from datetime import datetime
from typing import Iterator

import logging
import argparse
import re
import pathlib
import json
import itertools
from collections import deque
import subprocess

import rasterio
#from rasterio import merge
from rasterio.enums import Resampling
#from rasterio import merge
import merge
import numpy as np
from netCDF4 import Dataset, date2num
from tlz import concat, pluck

DEFAULT_BOUNDS = "-102 20 -58 48"
RES_LEVELS = ('core', 'storm', 'synoptic')
FNAME_RE = re.compile(rf"(?P<storm>\w+?)\.(?P<date>\d{{10}})\.hwrfprs\.(?P<level>{'|'.join(RES_LEVELS)})\.(?P<res>\d+?p\d+?)\.f(?P<fhour>\d+?)\.grb2.*?", flags=re.ASCII)

FILL_VALUE = np.float32(2147483647)
ATTRS = {
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
        "field": "lat, scalar, series"}
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
    "UGRD": "U2D",
    "VGRD": "V2D",
    "PRES": "PSFC",
    "TMP": "T2D",
    "SPFH": "Q2D",
    "DLWRF": "LWDOWN",
    "DSWRF": "SWDOWN",
    "PRATE": "RAINRATE"
}


class NCWriter:
    def __init__(self, filename, compress=True):
        """Open a dataset at the path given by filename for writing.

        Args:
            filename (pathlib.Path): file path to write to
            compress (bool, optional): use compression for variables. Defaults to True.
        """
        self.filename = filename
        self._handle = Dataset(filename, 'w')
        self.compress = compress

    def create_coordinate(self, name, size, dtype, attrs):
        """Create a coordinate in the netCDF file.

        Args:
            name (str): name of coordinate
            size (int): size of coordinate
            dtype (str): data type of coordinate
            attrs (dict): attributes for coordinate
        """
        assert self._handle.isopen()
        self._handle.createDimension(name, size)
        v = self._handle.createVariable(name, dtype, dimensions=(name,), zlib=self.compress)
        v.setncatts(attrs)

    def create_variable(self, name, dtype, dims, attrs):
        """Create a variable in the netCDF file.

        Args:
            name (str): name of variable
            dtype (str): data type of variable
            dims (tuple): dimensions of variable
            attrs (dict): attributes for variable
        """
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


def validate_recipe(recipe):
    p = recipe['priority']

    for ts, v in p.items():
        for src in v:
            if ts not in recipe[src]["timesteps"]:
                raise KeyError(f"{src} missing timestep {ts}")
            
    for k in sources(p):
        for files in recipe[k]["timesteps"].values():
            for f in concat(files):
                if not pathlib.Path(f).is_file():
                    raise ValueError(f"{f} is not a file")

def sources(priority):
    return set(concat(priority.values()))

def sort_dictionary_by_key(d):
    """Sort a dictionary by key.
    Works because dictionaries in Python 3.7+ retain insertion order.

    Args:
        d (dict): dictionary

    Returns:
        dict: new sorted dictionary
    """
    return dict(sorted(d.items()))

def read_recipe(path):
    with open(path) as fin:
        rv = json.load(fin)

    # Ensure timesteps are sorted
    rv['priority'] = sort_dictionary_by_key(rv['priority'])
    for src in sources(rv['priority']):
        rv[src]["timesteps"] = sort_dictionary_by_key(rv[src]["timesteps"])
        
    return rv


def list_wrap(obj):
    if not isinstance(obj, list):
        raise ValueError("Object must be a list")
    if len(obj) > 0 and isinstance(obj[0], list):
        return obj
    return [obj]


def combine(*sources):
    """This employs a bit of magic to get the sequences to align just right.
    Given all the sources for a timestep, yield frames."""
    return map(concat, itertools.product(*map(list_wrap, sources)))


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
    dt = kwargs.pop('dtype', 'float32')
    bounds = kwargs.pop('bounds', None)
    rv, t = merge.merge(res_levels,
                        bounds=bounds,
                        nodata=np.nan,
                        dtype=dt,
                        res=res,
                        indexes=indices,
                        resampling=Resampling.bilinear)
    return rv, t


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
    parser.add_argument('recipe', type=pathlib.Path, help="Recipe to execute")
    parser.add_argument('output_path', type=pathlib.Path, help='Directory to store processed files')
    parser.add_argument('-b', '--bounds', default=DEFAULT_BOUNDS, help="boundary points (left,bottom,right,top)")
    args = parser.parse_args()

    # Basic validation for ramp-weights
    global BOUNDS
    BOUNDS = args.bounds = tuple(map(float, args.bounds.split()))
    return args

def main(args):
    recipe = read_recipe(args.recipe)
    validate_recipe(recipe)
    priority = recipe['priority']

    BUFFER = deque()

    NCFile = NCWriter(args.output_path.joinpath("Output.nc"), compress=True)
    NCFile.create_coordinate("time", None, 'f8', ATTRS['time'])
    ref_time = ATTRS['time']['units']
    indices=None

    weights = None
    frames = []
    for ts in sorted(priority.keys()):
        order = priority[ts]
        _sources = pluck(ts, (recipe[src]["timesteps"] for src in order))
        frames.append(tuple(combine(*_sources)))

    # Merging and blending operation
    # For each time step we build a the following sequence
    # ts = [[n], [[a, b, c], [x, y, z]], [m]]
    # We generate the frames:
    # frame1 = [n, a, b, c, m]
    # frame2 = [n, x, y, z, m]
    # We merge each frame. Then blend (if there are two frames)


    for idx in enumerate(frames):
        for frame in frames:
            copy_grib(frame)
            ds_files = tuple(rasterio.open, frame)
            res = min(_x.res for _x in ds_files)
            X = merge_layers(ds_files, res=res, bounds=args.bounds)
            print("Shape:", X[0].shape)
            X += (indices,)  # add indices to tuple
            BUFFER.append(X)

        if len(BUFFER) > 1:
            # Determine length of blend by looking ahead
            if weights is None:
                idx_stop = idx + 1
                while idx_stop < len(frames):
                    if len(frames[idx_stop]) == 1:
                        break
                    idx_stop += 1
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
            NCFile.variables[vto][idx, ...] = rv[il]


        assert len(BUFFER) == 0


if __name__ == "__main__":
    logging.basicConfig()
    args = get_args()
    main(args)
