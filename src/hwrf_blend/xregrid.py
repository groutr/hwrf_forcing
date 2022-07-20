import argparse
import pathlib
import time
import json

import xesmf as xe
import xarray as xr
import numpy as np
import cftime

import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

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

REGRIDDER = None

def worker_load_weights(src, dst, weights, force_recalc):
    global REGRIDDER
        
    if weights and weights.is_file() and not force_recalc:
        print("Worker loading weights...")
        REGRIDDER = xe.Regridder(src, dst, 'bilinear', reuse_weights=True, weights=weights)
    else:
        print("Worker computing weights...")
        REGRIDDER = xe.Regridder(src, dst, 'bilinear', reuse_weights=False, filename=weights)


def date_ceil_hr(dt):
    """Ceil a datetime object."""
    if dt.minute == 0:
        return dt

    if dt.hour == 23:
        day = dt.day + 1
    else:
        day = dt.day
    return cftime.datetime(dt.year, dt.month, day, (dt.hour+1) % 24)


def regrid(forcing_file, nwm_grid, weights, output_path, ncatts=None):
    """Regrid the forcing_file and write to output_dir

    weights is the path to the precomputed regridding weights
    """

    enc = {'zlib': True}
    with xr.open_dataset(forcing_file, decode_cf=True, mask_and_scale=True, use_cftime=True, decode_times=True) as ds:
        print("Regridding", forcing_file)
        rv = REGRIDDER(ds, keep_attrs=True)
        #rv = rv.rename(VARMAP)
        rv = rv.rename({'x': "west_east", 'y': "south_north"})
        if ncatts:
            with open(ncatts) as fh:
                for k, v in json.load(fh).items():
                    rv[k].attrs.update(v)
        timestamp = date_ceil_hr(ds.time.values[0])
        # Set nan values to 0
        rv = rv.fillna(0)
        output_name = output_path / timestamp.strftime("%Y%m%d%H.LDASIN_DOMAIN1")
        print("Writing", output_name)
        rv.to_netcdf(output_name, encoding={v: enc for v in rv.variables})
    return output_name

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

    parser.add_argument('--pool', type=int, default=1,
                        help="How many processes to use for parallelization")
    parser.add_argument('--ncatts', type=pathlib.Path, default=None, help="NetCDF attributes to attach to output")
    parser.add_argument('--output', type=pathlib.Path, default='.', help="Directory to write output")
    parser.add_argument('--weights', type=pathlib.Path, help="Precomputed weights")
    parser.add_argument('--recalc', action='store_true', help='Force recalculation of weights')
    args = parser.parse_args()
    if args.weights:
        args.weights = args.weights.resolve()
    args.output = args.output.resolve()
    return args

def main(args):
    weights = args.weights

    pargs = []
    for fn in sorted(args.forcings.glob("*.nc")):
        output = args.output
        pargs.append((fn, args.nwm_grid, weights, output, args.ncatts))
    piargs = iter(pargs)

    src_grd = get_masterblend_grid(fn)
    dst_grd = get_nwm_grid(args.nwm_grid)

    ncpus = min(cpu_count(), args.pool)
    with ProcessPoolExecutor(max_workers=ncpus, initializer=worker_load_weights, initargs=(src_grd, dst_grd, weights, args.recalc)) as executor:
        print("Initialized pool with", ncpus, "processes")
        tasks = []
        for args in piargs:
            fut = executor.submit(regrid, *args)
            tasks.append(fut)

        for f in concurrent.futures.as_completed(tasks):
            print("Completed", f.result())

    print("Finished regridding all inputs")

if __name__ == "__main__":
    args = get_options()
    main(args)

