import argparse
import pathlib
import time
import json

import xesmf as xe
import xarray as xr
import numpy as np

from multiprocessing import Pool

VARMAP = {
    "uwnd": "U2D",
    "vwnd": "V2D",
    "DLWRF_surface": "LWDOWN",
    "DSWRF_surface": "SWDOWN",
    "PRES_surface": "PSFC",
    "TMP_2maboveground": "T2D",
    "SPFH_2maboveground": "Q2D",
    "PRATE_surface": "RAINRATE"
}

def regrid(forcing_file, nwm_grid, weights, output_path, ncatts=None):
    """Regrid the forcing_file and write to output_dir

    weights is the path to the precomputed regridding weights
    """
    src_grid = get_masterblend_grid(forcing_file)
    dst_grid = get_nwm_grid(nwm_grid)

    if weights.exists():
        print("Loading weights...")
        rg = xe.Regridder(src_grid, dst_grid, 'bilinear', reuse_weights=True, weights=weights)
    else:
        print("Computing weights...")
        rg = xe.Regridder(src_grid, dst_grid, 'bilinear', reuse_weights=False, filename=weights)
    enc = {'zlib': True}
    with xr.open_dataset(forcing_file) as ds:
        print("Regridding", forcing_file)
        rv = rg(ds)
        rv = rv.rename(VARMAP)
        if ncatts:
            with open(ncatts) as fh:
                for k, v in json.load(fh).items():
                    rv[k].attrs.update(v)
        print("Writing", output_path)
        rv.to_netcdf(output_path, encoding={v: enc for v in rv.variables})


def get_masterblend_grid(filename):
    with xr.open_dataset(filename) as ds:
        return {'lon': ds.longitude.values,
                'lat': ds.latitude.values}

def get_nwm_grid(filename):
    with xr.open_dataset(filename) as ds:
        ds = ds.sel({"Time": 0})
        return {'lon': ds.XLONG_M.values,
                'lat': ds.XLAT_M.values}

def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('forcings', type=pathlib.Path, help='Path of forcings inputs to regrid.')
    parser.add_argument('nwm_grid', type=pathlib.Path, help='Location of NWM grid.')

    parser.add_argument('--ncatts', type=pathlib.Path, default=None, help="NetCDF attributes to attach to output")
    parser.add_argument('--output', type=pathlib.Path, help="Directory to write output")
    parser.add_argument('--weights', type=pathlib.Path, help="Precomputed weights")
    args = parser.parse_args()
    args.weights = args.weights.resolve()
    args.output = args.output.resolve()
    return args

def main(args):
    weights = args.weights

    pargs = []
    for fn in args.forcings.glob("*.nc"):
        output = args.output / f"{fn.name}.regridded"
        pargs.append((fn, args.nwm_grid, weights, output, args.ncatts))
    piargs = iter(pargs)

    if not weights.exists():
        regrid(*next(piargs))

    with Pool(4) as pool:
        for task_args in piargs:
            pool.apply_async(regrid, args=task_args)

        pool.close()
        pool.join()

    print("Finished regridding all inputs")

if __name__ == "__main__":
    args = get_options()
    main(args)

