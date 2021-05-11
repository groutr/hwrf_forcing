from typing import Iterator

import logging
import argparse
import re
import datetime
import pathlib
import itertools
import functools
import operator
from collections import deque, defaultdict
from dataclasses import dataclass
import subprocess

import rasterio
#from rasterio import merge
from rasterio.enums import Resampling
#from rasterio import merge
import merge
import numpy as np
from netCDF4 import Dataset, num2date, date2num
from tlz import pluck, groupby

DEFAULT_BOUNDS = "-102 20 -58 48"
RES_LEVELS = ('core', 'storm', 'synoptic')
FNAME_RE = re.compile(rf"(?P<storm>\w+?)\.(?P<date>\d{{10}})\.hwrfprs\.(?P<level>{'|'.join(RES_LEVELS)})\.(?P<res>\d+?p\d+?)\.f(?P<fhour>\d+?)\.grb2.*?", flags=re.ASCII)

FILL_VALUE = np.float32(2147483647)
ATTRS = {
    "UGRD": {"name": "eastward_wind",
        "standard_name": "eastward_wind",
        "short_name": "u10",
        "level": "10 m above ground",
        "units": "m s-1",
        "field": "U, scalar, series",
        "long_name": "U wind component",
        "_FillValue": FILL_VALUE},
    "VGRD": {"units": "m s-1",
        "field": "V, scalar, series",
        "long_name": "V wind component",
        "name": "northward_wind",
        "standard_name": "northward_wind",
        "short_name": "v10",
        "level": "10 m above ground",
        "_FillValue": FILL_VALUE},
    "PRMSL": {"units": "Pa",
        "field": "P, scalar, series",
        "long_name": "Pressure Reduced to MSL",
        "name": "air_pressure",
        "standard_name": "air_pressure",
        "short_name": "msl",
        "level": "Mean sea level",
        "_FillValue": FILL_VALUE},
    "time": {"axis": "T",
        "conventions": "relative julian days with decimal part (as parts of the day)",
        "field": "time, scalar, series",
        "units": "days since 1990-01-01 00:00:00",
        "long_name": "julian day (UT)"},
    "longitude": {"short_name": "long",
        "standard_name": "longitude",
        "name": "longitude",
        "long_name": "longitude",
        "units": "degree_east",
        "field": "lon, scalar, series"},
    "latitude": {"long_name": "latitude",
        "short_name": "lat",
        "name": "latitude",
        "standard_name": "latitude",
        "units": "degree_north",
        "field": "lat, scalar, series"},
    "TMP": {"units": "K",
        "field": "TMP_2aboveground, scalar, series",
        "long_name": "2-meter air temperature",
        "name": "air_temperature",
        "standard_name": "air_temperature",
        "short_name": "T2D",
        "level": "2 m above ground",
        "_FillValue": FILL_VALUE},
    "SPFH": {"units": "kg/kg",
        "field": "SPFH_2maboveground, scalar, series",
        "long_name": "Specific humidity, dimensionless ratio of the mass of water vapor to the total mass of the system",
        "name": "specific_humidity",
        "standard_name": "specific_humidity",
        "short_name": "Q2D",
        "level": "2 m above ground",
        "_FillValue": FILL_VALUE},
    "PRATE": {"name": "rainfall",
        "standard_name": "rainfall_rate",
        "short_name": "PRATE",
        "level": "Surface",
        "units": "mm/s",
        "field": "PRATE_surface, scalar, series",
        "long_name": "Precipitation Rate at surface",
        "_FillValue": FILL_VALUE},
    "DLWRF": {"units": "W/m^2",
            "field": "DLWRF_surface, scalar, series",
            "long_name": "Surface downward long-wave radiation flux",
            "name": "surface_downward_longwave_flux",
            "standard_name": "surface_downward_longwave_flux",
            "short_name": "LWDOWN",
            "level": "Surface",
            "_FillValue": FILL_VALUE},
    "DSWRF": {"units": "W/m^2",
            "field": "DSWRF_surface, scalar, series",
            "long_name": "Surface downward short-wave radiation flux",
            "name": "surface_downward_shortwave_flux",
            "standard_name": "surface_downward_shortwave_flux",
            "short_name": "SWDOWN",
            "level": "Surface",
            "_FillValue": FILL_VALUE},
    "PRES": {"units": "Pa",
            "field": "PRES_surface, scalar, series",
            "long_name": "Surface pressure",
            "name": "surface_air_pressure",
            "standard_name": "surface_air_pressure",
            "short_name": "psfc",
            "level": "Surface",
            "_FillValue": FILL_VALUE}
}

GRIB_COPY = ['grib_copy', '-w', 'shortName=prmsl/10u/10v/sp/2sh/prate/2d/tcc/dlwrf/dswrf/2t,typeOfLevel=surface/meanSea/heightAboveGround,level=0/2/10']

PATTERNS = [{"GRIB_ELEMENT": "TMP", "GRIB_SHORT_NAME": "2-HTGL", "GRIB_PDS_PDTN": "0"},
            {"GRIB_ELEMENT": "PRMSL", "GRIB_SHORT_NAME": "0-MSL", "GRIB_PDS_PDTN": "0"},
            {"GRIB_ELEMENT": "UGRD", "GRIB_SHORT_NAME": "10-HTGL", "GRIB_PDS_PDTN": "0"},
            {"GRIB_ELEMENT": "VGRD", "GRIB_SHORT_NAME": "10-HTGL", "GRIB_PDS_PDTN": "0"},
            {"GRIB_ELEMENT": "PRES", "GRIB_SHORT_NAME": "0-SFC", "GRIB_PDS_PDTN": "0"},
            {"GRIB_ELEMENT": "DLWRF", "GRIB_SHORT_NAME": "0-SFC", "GRIB_PDS_PDTN": "0"},
            {"GRIB_ELEMENT": "DSWRF", "GRIB_SHORT_NAME": "0-SFC", "GRIB_PDS_PDTN": "0"},
            {"GRIB_ELEMENT": "SPFH", "GRIB_SHORT_NAME": "2-HTGL", "GRIB_PDS_PDTN": "0"},
            {"GRIB_ELEMENT": "PRATE", "GRIB_SHORT_NAME": "0-SFC", "GRIB_PDS_PDTN": "0"},
#            {"GRIB_ELEMENT": "DPT", "GRIB_SHORT_NAME": "2-HTGL", "GRIB_PDS_PDTN": "0"},
            {"GRIB_ELEMENT": "HGT", "GRIB_SHORT_NAME": "0-SFC", "GRIB_PDS_PDTN": "0"}]
NC_VARS = {"PRMSL": "msl",
    "UGRD": "u10",
    "VGRD": "v10",
    "PRES": "PSFC",
    "TMP": "T2D",
    "SPFH": "Q2D",
    "DLWRF": "LWDOWN",
    "DSWRF": "SWDOWN",
    "PRATE": "PRATE"
}


def parse_filename(filename:pathlib.Path):
    """
    Parse a filename for storm related information.

    Args:
        filename: Filename to parse

    Returns:
        (Filename)
        (None): Returned if filename cannot be parsed.
    """
    match = FNAME_RE.search(filename.name)
    if match:
        storm = match.group('storm')
        date = datetime.datetime.strptime(match.group('date'), '%Y%m%d%H')
        level = match.group('level')
        res = float(match.group('res').replace('p', '.'))
        fhour = int(match.group('fhour'))
        return Filename(filename, storm, date, level, res, fhour)


@dataclass
class Filename:
    filename: str
    storm: str
    date: datetime.datetime
    level: str
    res: float
    fhour: int

    @property
    def cycle(self):
        return self.date.hour

    def fdate(self):
        """
        Return the real forecasted date.
        """
        return self.date + datetime.timedelta(hours=self.fhour)


class DFlowNCWriter:
    def __init__(self, filename, compress=True):
        self.filename = filename
        self._handle = Dataset(filename, 'w')
        self.compress = compress

    def create_coordinate(self, name, size, dtype, attrs):
        assert self._handle.isopen()
        self._handle.createDimension(name, size)
        v = self._handle.createVariable(name, dtype, dimensions=(name,), zlib=self.compress)
        v.setncatts(attrs)

    def create_variable(self, name, dtype, dims, attrs):
        assert self._handle.isopen()
        v = self._handle.createVariable(name, dtype, dimensions=dims, zlib=self.compress)
        v.setncatts(attrs)

    @property
    def variables(self):
        assert self._handle.isopen()
        return self._handle.variables

    @property
    def dimensions(self):
        assert self._handle.isopen()
        return self._handle.dimensions

    def close(self):
        if self._handle.isopen():
            self._handle.close()


def copy_grib(files:Iterator, output_path:pathlib.Path, suffix:str='.new') -> list:
    """
    Run grib_copy on the grib inputs.
    This is required to work around a regression GDAL 3.1 which
    will fail to read certain types of grib files.

    As a bonus, this function will copy out only the variables needed.

    Args:
        files (Iterator): Candidate files to process
        suffix (str): Suffix for copied files. Defaults to .new

    Returns:
        (list): list of copied files

    Raises:
        CalledProcessError: Raised if the subprocess call failed
    """
    rv_files = []
    for f in files:
        output = output_path / (f.name + suffix)
        if not output.exists():
            print("Preprocessing", f)
            subprocess.run(GRIB_COPY + [f, output], check=True)
        else:
            print("Skipping existing file", output)
        rv_files.append(output)
    return rv_files


def order_layers(layers):
    """
    Order layers according to RES_LEVELS
    """
    ordered = []
    for r in RES_LEVELS:
        for L in layers:
            if r in L.name:
                ordered.append(L)
                break
    return ordered

def complement_re(filename):
    """Generate a regular expression to match
    the complementary resolution levels.

    This assumes that the filenames are regular.

    Args:
        filename (str): filename to find complements of

    Returns:
        (str): Regular expression to match the different resolution
        levels of the same time step.
    """
    parts = filename.name.split('.')

    rlvls = '|'.join(RES_LEVELS)
    try:
        del parts[3]  # remove core/storm/synoptic
        del parts[3]  # remove resolution level
    except IndexError:
        print(parts)
        raise
    parts.insert(3, r"(\d+?p\d+?)")
    parts.insert(3, f"({rlvls})")
    return f"^{'.'.join(parts)}$"


def discover_directory(path, glob="*.*.hwrfprs.*.*.*"):
    """Discover resolution levels for each timestep.

    Args:
        path (str, pathlib.PathLike): Path to directory
        glob (str): Glob to filter directory contents

    Returns:
        (List[List[pathlib.PathLike]]): A list of associated resolution levels.
    """
    path = pathlib.Path(path)

    if glob:
        fileset = tuple(path.glob(glob))
    else:
        fileset = tuple(path.iterdir())

    seen = set()
    pairs = []
    for fn in fileset:
        if fn in seen:
            continue
        p = []
        pat = complement_re(fn)
        for t in fileset:
            if re.match(pat, t.name):
                p.append(t)
                seen.add(t)
        if len(p) != len(RES_LEVELS):
            print(p, "Missing a resolution level")
        else:
            pairs.append(order_layers(p))
    return pairs

def match_case(case, cases):
    for c in cases:
        for k, v in c.items():
            if k in case and case[k] == v:
                continue
            else:
                break
        else:
            return c
    else:
        return None

def find_vars(ds):
    indices = {}
    for v in range(1, ds.count+1):
        var = match_case(ds.tags(v), PATTERNS)
        if var:
            indices[v] = ds.tags(v)
    return indices

def celsius_to_kelvin(arr):
    """Convert Celsius temps to Kelvin"""
    return arr + 273.15

def merge_layers(res_levels, res=None, indices=None, **kwargs):
    ds_readers = [isinstance(rl, rasterio.DatasetReader) for rl in res_levels]
    if all(ds_readers):
        X =_merge_layers(res_levels, res=res, indices=indices, **kwargs)
    elif not any(ds_readers):
        _res_levels = [rasterio.open(rl) for rl in res_levels]
        X = _merge_layers(_res_levels, res=res, indices=indices, **kwargs)

        # close objects
        for rl in _res_levels:
            rl.close()
    else:
        raise ValueError("Only all strings or all rasterio dataset objects required")
    return X


def _merge_layers(res_levels, res=None, indices=None, **kwargs):
    """
    Merge layers together using Rasterio.

    Arguments:
        res_levels: Ra
        indices (list[int]): Indices to process.
            If None, all layers are merged (MEMORY INTENSIVE!!)

    Returns:
        rv: Merged layers (3D array).
            Order follows order of variables in dataset or order given in indices if specified.
        t: transform for merged dataset
    """
    nodata_vals = [d.nodata for d in res_levels]
    rv, t = merge.merge(res_levels,
                        bounds=BOUNDS,
                        nodata=np.nan,
                        dtype='float32',
                        res=res,
                        indexes=indices,
                        resampling=Resampling.bilinear)
    # handle possibly different no_data values
    for i, nd in enumerate(nodata_vals):
        if nd is not None:
            mask = rv == nd
            rv[mask] = np.nan

    return rv, t

def export_dflow_nc(export_path, date, rv, transform, layers, compress=True):
    """
    Write outputs to a netcdf file.

    Arguments:
        export_file: filename
        longitude: longitude coordinate array
        latitude: latitude coordinate array
        variables: dictonary of variables to write {string: data}
        compress: compress variables
    """
    filename = f"{date.strftime('%Y%m%d%H')}_LDASIN"
    efn = pathlib.Path(export_path, filename)
    print("Writing output to:", efn)
    nc = Dataset(str(efn), "w")

    # Dimensions
    timed = nc.createDimension("time", 1)
    longd = nc.createDimension("longitude", rv.shape[2])
    latd = nc.createDimension("latitude", rv.shape[1])

    # Variables
    timev = nc.createVariable("time", 'f8', dimensions=("time",), zlib=compress)
    longv = nc.createVariable("longitude", 'f8', dimensions=("longitude",), zlib=compress)
    latv = nc.createVariable("latitude", 'f8', dimensions=("latitude",), zlib=compress)
    longv.setncatts(ATTRS["longitude"])
    latv.setncatts(ATTRS["latitude"])
    timev.setncatts(ATTRS["time"])

    longitude = rasterio.transform.xy(transform, range(rv.shape[2]), [0]*rv.shape[2])
    latitude = rasterio.transform.xy(transform, [0]*rv.shape[1], range(rv.shape[1]))
    longv[:] = np.array(longitude[0])
    latv[:] = np.array(latitude[1])
    timev[:] = date2num(date, "days since 1990-01-01 00:00:00", calendar='julian')

    for i, vname in enumerate(layers):
        vto = NC_VARS[vname]
        v = nc.createVariable(vto, 'f4', dimensions=("time", "longitude", "latitude"), zlib=compress)
        v.setncatts(ATTRS[vname])

        v[:] = rv[i].T

    nc.close()

def linear_blend(length):
    step = 1/(length + 1)
    val = step
    while val < 1:
        yield val
        val += step

def get_lons_lats(transform, shape):
    lats = rasterio.transform.xy(transform, range(shape[1]), [0] * shape[1])
    lons = rasterio.transform.xy(transform, [0] * shape[2], range(shape[2]))
    return np.asarray(lons[0]), np.asarray(lats[1])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=pathlib.Path, help="directory to process")
    parser.add_argument('output_path', type=pathlib.Path, help='Directory to store processed files')
    parser.add_argument('--storm', required=False, help="only process a particular storm")
    parser.add_argument('-b', '--bounds', default=DEFAULT_BOUNDS, help="boundary points (left,bottom,right,top)")
    args = parser.parse_args()

    # Basic validation for ramp-weights
    global BOUNDS
    BOUNDS = args.bounds = tuple(map(float, args.bounds.split()))
    return args

def main(args):
    input_files = copy_grib(args.path.glob('*.hwrfprs.*.grb2'), args.output_path)
    storm_data = discover_directory(args.output_path, glob="*.hwrfprs.*.grb2.new")

    # Index the timesteps in the directory
    time_steps = defaultdict(list)
    for f in storm_data:
        pf = [parse_filename(x) for x in f]
        time_steps[pf[0].fdate()].append(pf)

    BUFFER = deque()

    NCFile = DFlowNCWriter(args.output_path.joinpath(args.storm + ".nc"), compress=True)
    NCFile.create_coordinate("time", None, 'f8', ATTRS['time'])
    ref_time = ATTRS['time']['units']
    indices=None

    sorted_times = sorted(time_steps.keys())
    weights = None
    for idx, ts in enumerate(sorted_times):
        print("Processing timestep:", ts)
        # merge
        print("Merging...")
        for t in sorted(time_steps[ts], key=lambda x: x[0].date):
            res = min(_x.res for _x in t)
            ds_files = [_x.filename for _x in t]
            with rasterio.open(ds_files[0]) as ds:
                indices = find_vars(ds)
            X = merge_layers(ds_files, res=res, indices=list(indices.keys()))
            print("Shape:", X[0].shape)
            X += (indices,)  # add indices to tuple
            BUFFER.append(X)

        if len(BUFFER) > 1:
            # Determine length of blend by looking ahead
            if weights is None:
                idx_stop = idx + 1
                while idx_stop < len(sorted_times):
                    key = sorted_times[idx_stop]
                    if len(time_steps[key]) == 1:
                        break
                weights = linear_blend(idx_stop - idx)

            # blend
            print("Blending buffer...")
            assert len(BUFFER) == 2

            weight = next(weights)
            X1 = BUFFER.popleft()
            X2 = BUFFER.popleft()
            print(f"Blending with weight of {1-weight}X1 + {weight}X2")
            _x1 = X1[0]
            _x2 = X2[0]
            # Do (1-weight) * X1 + weight * X2 without intermediate arrays
            np.multiply(_x1, (1-weight), _x1)
            np.multiply(_x2, weight, _x2)
            np.add(_x1, _x2, _x1)
            BUFFER.appendleft((_x1, *X1[1:]))
        else:
            weights = None

        rv, transform, layers = BUFFER.popleft()
        # Write outputs
        if idx == 0:
            NCFile.create_coordinate("longitude", rv.shape[2], 'f4', ATTRS["longitude"])
            NCFile.create_coordinate("latitude", rv.shape[1], 'f4', ATTRS["latitude"])
            NCFile._handle.setncattr("transform", np.asarray(transform.to_gdal()))
            lons, lats = get_lons_lats(transform, rv.shape)
            NCFile.variables["longitude"][:] = lons
            NCFile.variables["latitude"][:] = lats
            #NCFile.variables["longitude"][:] = np.linspace(BOUNDS[0], BOUNDS[2], rv.shape[2])
            #NCFile.variables["latitude"][:] = np.linspace(BOUNDS[1], BOUNDS[3], rv.shape[1])

            for layer in (v["GRIB_ELEMENT"] for v in layers.values()):
                vto = NC_VARS[layer]
                NCFile.create_variable(vto, 'f4', ("time", "latitude", "longitude"), ATTRS[layer])

        NCFile.variables["time"][idx] = date2num(ts, ref_time, calendar='julian')
        for il, layer in enumerate(layers.values()):
            element = layer["GRIB_ELEMENT"]
            vto = NC_VARS[element]
            if layer["GRIB_UNIT"] == "[C]" and ATTRS[element]["units"] == "K":
                rv[il] = celsius_to_kelvin(rv[il])
            NCFile.variables[vto][idx, ...] = rv[il]


        assert len(BUFFER) == 0


if __name__ == "__main__":
    logging.basicConfig()
    args = get_args()
    main(args)
