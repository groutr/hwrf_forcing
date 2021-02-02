import argparse
import pathlib

import ESMF
import xarray as xr
import numpy as np


def make_lat_lon_grid(lat, lon):
    if lat.dtype == np.float32 or lon.dtype == np.float32:
        typekind = ESMF.TypeKind.R4
    else:
        typekind = ESMF.TypeKind.R8

    G = ESMF.Grid(np.array(lat.shape),
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

    args = parser.parse_args()
    return args

def main(args):
    nwm_grid = get_nwm_grid(args.nwm_grid)
    dst_grid = make_lat_lon_grid(*nwm_grid)

    with xr.open_dataset(args.forcings) as forcings:
        flats = forcings['latitude'][:]
        flons = forcings['longitude'][:]
        src_grid = make_lat_lon_grid(flats, flons)

        srcF, dstF = make_fields(src_grid, dst_grid)

        regridder = ESMF.Regrid(srcF, dstF, src_mask_values=np.array([0]),
                regrid_method=ESMF.RegridMethod.BILINEAR, unmapped_action=ESMF.UnmappedAction.IGNORE,
                filename=filename, rh_filename=filename+'.routehandle')



if __name__ == "__main__":
    args = get_options()

