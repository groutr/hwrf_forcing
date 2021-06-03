import argparse
import pathlib
import time
from multiprocessing import Pool

import ESMF
import xarray as xr
import numpy as np
import netCDF4

def make_lat_lon_grid(lat, lon):
    if lat.dtype == np.float32 or lon.dtype == np.float32:
        typekind = ESMF.TypeKind.R4
    else:
        typekind = ESMF.TypeKind.R8

    if lat.ndim == 1 and lon.ndim == 1:
        shape = np.array([lat.size, lon.size])
        lat = np.atleast_2d(lat).T
    else:
        shape = np.array(lat.shape)

    print("Creating grid of shape", shape)
    G = ESMF.Grid(shape,
            coord_sys=ESMF.CoordSys.SPH_DEG,
            coord_typekind=typekind,
            staggerloc=ESMF.StaggerLoc.CENTER)

    LO = G.get_coords(0)
    LA = G.get_coords(1)
    LO[:] = lon
    LA[:] = lat
    return G

# Create source grid and destination grid
def make_fields(src_grid, dst_grid):
    srfF = ESMF.Field(src_grid, name="SOURCE")
    dstF = ESMF.Field(dst_grid, name="DEST")
    return srfF, dstF


def make_regridder(src, dst, weight_filename):
    if weight_filename.exists():
        print("Creating regridder from file")
        rv = ESMF.RegridFromFile(src, dst,
                filename=str(weight_filename))
    else:
        print("Creating regridder")
        rv = ESMF.Regrid(src, dst,
                filename=str(weight_filename),
                src_mask_values=np.array([np.nan, 0]),
                dst_mask_values=np.array([np.nan, 0]),
                regrid_method=ESMF.RegridMethod.BILINEAR,
                unmapped_action=ESMF.UnmappedAction.IGNORE)
    return rv

def get_blend_grid(filename):
    with xr.open_dataset(filename) as ds:
        ds = ds.sel({"time": 0})
        xlat = ds.latitude.values
        xlong = ds.longitude.values
    return np.meshgrid(xlong, xlat, indexing='ij')

def get_nwm_grid(filename):
    with xr.open_dataset(filename) as ds:
        ds = ds.sel({"Time": 0})
        xlat = ds.XLAT_M.values
        xlong = ds.XLONG_M.values
    return xlat, xlong


def regrid(src, dst, weights, nwm_file):
    nwm_grid = get_nwm_grid(nwm_file)
    dst_grid = make_lat_lon_grid(*nwm_grid)

    with xr.open_dataset(src, decode_times=False) as ds:
        flats = ds['latitude'].values
        flons = ds['longitude'].values
        src_grid = make_lat_lon_grid(flats, flons)

        srcF, dstF = make_fields(src_grid, dst_grid)
        start = time.perf_counter()
        regridder = make_regridder(srcF, dstF, weights)
        print("Regridder created", time.perf_counter() - start)

        clon = dstF.grid.get_coords(0)
        clat = dstF.grid.get_coords(1)
        rf = netCDF4.Dataset(dst, mode='w')
        rf.createDimension('time', len(ds.time))
        v = rf.createVariable('time', 'f8', dimensions=('time',), zlib=True)
        v[:] = ds.time.values
        v.setncatts(ds.time.attrs)
        rf.createDimension('x', clon.shape[0])
        rf.createDimension('y', clon.shape[1])
        v = rf.createVariable('latitude', 'f8', dimensions=('x', 'y'), zlib=True)
        v[:] = clat
        v = rf.createVariable('longitude', 'f8', dimensions=('x', 'y'), zlib=True)
        v[:] = clon

        tmp = np.full((len(ds.time), *clon.shape), np.nan, dtype='float32')
        for V in ds.data_vars:
            print("Regridding", V)
            fv = ds[V]
            rv = rf.createVariable(V, 'f4', dimensions=('time', 'x', 'y'), zlib=True)
            tmp.fill(np.nan)
            for ts in range(ds.time.size):
                srcF.data[:] = fv[ts].values
                regridder(srcF, dstF)
                tmp[ts, ...] = dstF.data[:]
            rv[:] = tmp
        rf.close()
        regridder.destroy()
    print("Regridding", src, "finished")

def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('forcings', type=pathlib.Path, help='Location of forcings files')
    parser.add_argument('nwm_grid', type=pathlib.Path, help='Location of NWM grid')

    parser.add_argument('--weights', type=pathlib.Path, help="Precomputed weights")
    parser.add_argument('--output', type=pathlib.Path, help="Directory to put output")
    args = parser.parse_args()

    args.weights = args.weights.resolve()
    args.output = args.output.resolve()
    return args

def main(args):

    # we want to perform first regridding to calculate weights
    pargs = []
    for fn in args.forcings.glob("*.nc"):
        pargs.append((fn, args.output.joinpath(f"{fn.name}.regridded"),
                    args.weights, args.nwm_grid))

    piargs = iter(pargs)
    if not args.weights.exists():
        regrid(*next(piargs))

    with Pool(4) as pool:
        pool.starmap_async(regrid, piargs)
        pool.close()
        print("Waiting for all processes to finish...")
        pool.join()
    print("Regridding task finished")


if __name__ == "__main__":
    args = get_options()
    main(args)

