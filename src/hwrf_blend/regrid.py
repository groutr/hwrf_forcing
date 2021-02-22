import argparse
import pathlib
import time

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
            staggerloc=ESMF.StaggerLoc.CENTER,
            pole_kind=np.array([ESMF.PoleKind.MONOPOLE]*2))

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


def get_nwm_grid(filename):
    with xr.open_dataset(filename) as ds:
        ds = ds.sel({"Time": 0})
        xlat = ds.XLAT_M.values
        xlong = ds.XLONG_M.values
    return xlat, xlong

def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('forcings', type=pathlib.Path, help='Location of forcings file')
    parser.add_argument('nwm_grid', type=pathlib.Path, help='Location of NWM grid')

    methods = tuple(ESMF.api.constants.RegridMethod.__members__.keys())
    parser.add_argument('--method', choices=methods, help='Regridding method')
    parser.add_argument('--filename', help='filename for ESMF weights cache')
    args = parser.parse_args()
    return args

def main(args):
    nwm_grid = get_nwm_grid(args.nwm_grid)
    dst_grid = make_lat_lon_grid(*nwm_grid)

    with xr.open_dataset(args.forcings, decode_times=False) as forcings:
        flats = forcings['latitude'][:].values
        flons = forcings['longitude'][:].values
        src_grid = make_lat_lon_grid(flats, flons)

        print("Creating fields")
        srcF, dstF = make_fields(src_grid, dst_grid)

        print("Creating ESMF regridding object")
        filename = args.forcings.with_suffix('.esmf_weights')
        rh_filename = filename.with_suffix('.routehandle')
        start = time.perf_counter()
        regridder = ESMF.Regrid(srcF, dstF, src_mask_values=np.array([np.nan]),
                regrid_method=ESMF.RegridMethod.BILINEAR, unmapped_action=ESMF.UnmappedAction.IGNORE,
                rh_filename=str(rh_filename))
        print("Regridding took:", time.perf_counter()-start)

        clon = dstF.grid.get_coords(0)
        clat = dstF.grid.get_coords(1)
        rf = netCDF4.Dataset(str(filename.with_suffix(".regridded.nc")), mode='w')
        rf.createDimension('time', len(forcings.time))
        v = rf.createVariable('time', 'f8', dimensions=('time',), zlib=True)
        v[:] = forcings.time.values
        v.setncatts(forcings.time.attrs)
        rf.createDimension('x', clon.shape[0])
        rf.createDimension('y', clon.shape[1])
        v = rf.createVariable('latitude', 'f8', dimensions=('x', 'y'), zlib=True)
        v[:] = clat
        v = rf.createVariable('longitude', 'f8', dimensions=('x', 'y'), zlib=True)
        v[:] = clon

        tmp = np.full((len(forcings.time), *clon.shape), np.nan, dtype='float32')
        for V in forcings.data_vars:
            print("Regridding", V)
            fv = forcings[V]
            rv = rf.createVariable(V, 'f4', dimensions=('time', 'x', 'y'), zlib=True)
            tmp.fill(np.nan)
            for ts in range(forcings.time.size):
                print("timestep", ts, end='\r')
                srcF.data[:] = fv[ts].values
                regridder(srcF, dstF)
                tmp[ts, ...] = dstF.data[:]
            rv[:] = tmp
        rf.close()


if __name__ == "__main__":
    args = get_options()
    main(args)

