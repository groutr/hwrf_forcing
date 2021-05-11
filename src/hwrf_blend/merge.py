"""Copy valid pixels from input files to an output file.
This code was originally copied from rasterio project and modified to fix several shortcomings.
These improvements have been merged upstream and should be available in the rasterio 1.2.1 release.

The code below is under the following license:
-----------------------------------------------------
Copyright (c) 2013, MapBox
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of Mapbox nor the names of its contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-------------------------------------------------------
"""

from contextlib import contextmanager
import logging
import math
from pathlib import Path
import sys
import warnings
from collections.abc import Iterable

import numpy as np

import rasterio
from rasterio.enums import Resampling
from rasterio import windows
from rasterio.transform import Affine
from rasterio.coords import BoundingBox

logger = logging.getLogger(__name__)


def copy_first(old_data, new_data, old_nodata, new_nodata, **kwargs):
    mask = np.empty_like(old_data, dtype='bool')
    np.logical_not(new_nodata, out=mask)
    np.logical_and(old_nodata, mask, out=mask)
    np.copyto(old_data, new_data, where=mask)

MERGE_METHODS = {'first': copy_first}

def merge(
    datasets,
    bounds=None,
    res=None,
    nodata=None,
    dtype=None,
    precision=10,
    indexes=None,
    output_count=None,
    resampling=Resampling.nearest,
    method="first",
    dst_path=None,
    dst_kwds=None,
):
    """Copy valid pixels from input files to an output file.
    All files must have the same number of bands, data type, and
    coordinate reference system.
    Input files are merged in their listed order using the reverse
    painter's algorithm (default) or another method. If the output file exists,
    its values will be overwritten by input values.
    Geospatial bounds and resolution of a new output file in the
    units of the input file coordinate reference system may be provided
    and are otherwise taken from the first input file.
    Parameters
    ----------
    datasets : list of dataset objects opened in 'r' mode, filenames or pathlib.Path objects
        source datasets to be merged.
    bounds: tuple, optional
        Bounds of the output image (left, bottom, right, top).
        If not set, bounds are determined from bounds of input rasters.
    res: tuple, optional
        Output resolution in units of coordinate reference system. If not set,
        the resolution of the first raster is used. If a single value is passed,
        output pixels will be square.
    nodata: float, optional
        nodata value to use in output file. If not set, uses the nodata value
        in the first input raster.
    dtype: numpy dtype or string
        dtype to use in outputfile. If not set, uses the dtype value in the
        first input raster.
    precision: float, optional
        Number of decimal points of precision when computing inverse transform.
    indexes : list of ints or a single int, optional
        bands to read and merge
    output_count: int, optional
        If using callable it may be useful to have additional bands in the output
        in addition to the indexes specified for read
    resampling : Resampling, optional
        Resampling algorithm used when reading input files.
        Default: `Resampling.nearest`.
    method : str or callable
        pre-defined method:
            first: reverse painting
            last: paint valid new on top of existing
            min: pixel-wise min of existing and new
            max: pixel-wise max of existing and new
        or custom callable with signature:
        def function(old_data, new_data, old_nodata, new_nodata, index=None, roff=None, coff=None):
            Parameters
            ----------
            old_data : array_like
                array to update with new_data
            new_data : array_like
                data to merge
                same shape as old_data
            old_nodata, new_data : array_like
                boolean masks where old/new data is nodata
                same shape as old_data
            index: int
                index of the current dataset within the merged dataset collection
            roff: int
                row offset in base array
            coff: int
                column offset in base array
    dst_path : str or Pathlike, optional
        Path of output dataset
    dst_kwds : dict, optional
        Dictionary of creation options and other paramters that will be
        overlaid on the profile of the output dataset.
    Returns
    -------
    tuple
        Two elements:
            dest: numpy ndarray
                Contents of all input rasters in single array
            out_transform: affine.Affine()
                Information for mapping pixel coordinates in `dest` to another
                coordinate system
    """
    if method in MERGE_METHODS:
        copyto = MERGE_METHODS[method]
    elif callable(method):
        copyto = method
    else:
        raise ValueError('Unknown method {0}, must be one of {1} or callable'
                         .format(method, MERGE_METHODS))

    # Create a dataset_opener object to use in several places in this function.
    if isinstance(datasets[0], str) or isinstance(datasets[0], Path):
        dataset_opener = rasterio.open
    else:

        @contextmanager
        def nullcontext(obj):
            try:
                yield obj
            finally:
                pass

        dataset_opener = nullcontext

    with dataset_opener(datasets[0]) as first:
        first_profile = first.profile
        first_res = first.res
        first_bounds = first.bounds
        nodataval = first.nodatavals[0]
        dt = first.dtypes[0]

        if indexes is None:
            src_count = first.count
        elif isinstance(indexes, int):
            src_count = indexes
        else:
            src_count = len(indexes)

        try:
            first_colormap = first.colormap(1)
        except ValueError:
            first_colormap = None

    if not output_count:
        output_count = src_count

    # Extent from option or extent of all inputs
    if bounds:
        dst_w, dst_s, dst_e, dst_n = bounds
    else:
        # scan input files
        dst_w, dst_s, dst_e, dst_n = first_bounds
        for dataset in datasets:
            with dataset_opener(dataset) as src:
                left, bottom, right, top = src.bounds
            dst_w = min(dst_w, left)
            dst_s = min(dst_s, bottom)
            dst_e = max(dst_e, right)
            dst_n = max(dst_n, top)

    logger.debug("Output bounds: %r", (dst_w, dst_s, dst_e, dst_n))
    output_transform = Affine.translation(dst_w, dst_n)
    logger.debug("Output transform, before scaling: %r", output_transform)

    # Resolution/pixel size
    if not res:
        res = first_res
    elif not np.iterable(res):
        res = (res, res)
    elif len(res) == 1:
        res = (res[0], res[0])
    output_transform *= Affine.scale(res[0], -res[1])
    logger.debug("Output transform, after scaling: %r", output_transform)

    # Compute output array shape. We guarantee it will cover the output
    # bounds completely
    output_width = int(math.ceil((dst_e - dst_w) / res[0]))
    output_height = int(math.ceil((dst_n - dst_s) / res[1]))

    # Adjust bounds to fit
    dst_e, dst_s = output_transform * (output_width, output_height)
    logger.debug("Output width: %d, height: %d", output_width, output_height)
    logger.debug("Adjusted bounds: %r", (dst_w, dst_s, dst_e, dst_n))

    if dtype is not None:
        dt = dtype
        logger.debug("Set dtype: %s", dt)

    out_profile = first_profile
    out_profile.update(**(dst_kwds or {}))

    out_profile["transform"] = output_transform
    out_profile["height"] = output_height
    out_profile["width"] = output_width
    out_profile["count"] = output_count
    if nodata is not None:
        out_profile["nodata"] = nodata
        nodataval = nodata
        logger.debug("Set nodataval: %r", nodataval)

    fillval = 0
    if nodataval is not None:
        # Only fill if the nodataval is within dtype's range
        inrange = False
        if np.issubdtype(dt, np.integer):
            info = np.iinfo(dt)
            if info.min <= nodataval <= info.max:
                fillval = nodataval
                inrange = True
        elif np.issubdtype(dt, np.floating):
            if math.isnan(nodataval):
                fillval = nodataval
                inrange = True
            else:
                info = np.finfo(dt)
                if info.min <= nodataval <= info.max:
                    fillval = nodataval
                    inrange=True
        if not inrange:
            warnings.warn(
                "Input file's nodata value, %s, is beyond the valid "
                "range of its data type, %s. Consider overriding it "
                "using the --nodata option for better results." % (
                    nodataval, dt))
    else:
        nodataval = 0

    # create destination array
    dest = np.full((output_count, output_height, output_width), fillval, dtype=dt)

    for idx, dataset in enumerate(datasets):
        with dataset_opener(dataset) as src:
            # Real World (tm) use of boundless reads.
            # This approach uses the maximum amount of memory to solve the
            # problem. Making it more efficient is a TODO.

            # 1. Compute spatial intersection of destination and source
            try:
                common_bound = intersect_bounds(src.bounds, (dst_w, dst_s, dst_e, dst_n))
            except ValueError:
                continue


            # 2. Compute the source window
            src_window = from_bounds(
                *common_bound, transform=src.transform
            )
            logger.debug("Src %s window: %r", src.name, src_window)


            # 3. Compute the destination window
            dst_window = from_bounds(
                *common_bound, transform=output_transform
            )

            src_window = src_window.round_lengths(pixel_precision=0)
            dst_window = dst_window.round_lengths(pixel_precision=0)
            trows, tcols = dst_window.height, dst_window.width
            srows, scols = src_window.height, src_window.width

            # 4. Check to see if source overlaps with destination
            dst_window = dst_window.round_offsets(pixel_precision=0)
            trows, tcols = dst_window.height, dst_window.width
            roff, coff = dst_window.row_off, dst_window.col_off
            rstop = roff + trows
            cstop = coff + tcols
            if roff == rstop or coff == cstop:
                continue

            # 4. Read data in source window into temp
            # Check if resampling will happen
            if trows != srows or tcols != scols:
                print("Resampling expected!", f"({srows}, {scols}) -> ({trows}, {tcols})")

            temp_shape = (src_count, trows, tcols)
            temp = src.read(
                out_shape=temp_shape,
                window=src_window,
                boundless=False,
                masked=True,
                indexes=indexes,
                resampling=resampling,
            )

        # 5. Copy elements of temp into dest
        region = dest[:, roff:rstop, coff:cstop]
        if math.isnan(nodataval):
            region_nodata = np.isnan(region)
        else:
            region_nodata = region == nodataval
        temp_nodata = np.ma.getmaskarray(temp)

        copyto(region, temp, region_nodata, temp_nodata,
               index=idx, roff=roff, coff=coff)

    if dst_path is None:
        return dest, output_transform

    else:
        with rasterio.open(dst_path, "w", **out_profile) as dst:
            dst.write(dest)
            if first_colormap:
                dst.write_colormap(1, first_colormap)

def rowcol(transform, xs, ys, op=math.floor):
    """
    Returns the rows and cols of the pixels containing (x, y) given a
    coordinate reference system.
    Use an epsilon, magnitude determined by the precision parameter
    and sign determined by the op function:
        positive for floor, negative for ceil.
    Parameters
    ----------
    transform : Affine
        Coefficients mapping pixel coordinates to coordinate reference system.
    xs : list or float
        x values in coordinate reference system
    ys : list or float
        y values in coordinate reference system
    op : function
        Function to convert fractional pixels to whole numbers (floor, ceiling,
        round)
    Returns
    -------
    rows : list of ints
        list of row indices
    cols : list of ints
        list of column indices
    """

    if not isinstance(xs, Iterable):
        xs = [xs]
    if not isinstance(ys, Iterable):
        ys = [ys]

    eps = sys.float_info.epsilon
    if op(0.1) >= 1:
        eps = -eps

    invtransform = ~transform

    rows = []
    cols = []
    for x, y in zip(xs, ys):
        fcol, frow = invtransform * (x + eps, y + eps)
        cols.append(op(fcol))
        rows.append(op(frow))

    if len(cols) == 1:
        return rows[0], cols[0]
    return rows, cols

def from_bounds(left, bottom, right, top, transform=None,
                height=None, width=None):
    """Get the window corresponding to the bounding coordinates.
    Parameters
    ----------
    left: float, required
        Left (west) bounding coordinates
    bottom: float, required
        Bottom (south) bounding coordinates
    right: float, required
        Right (east) bounding coordinates
    top: float, required
        Top (north) bounding coordinates
    transform: Affine, required
        Affine transform matrix.
    height: int, required
        Number of rows of the window.
    width: int, required
        Number of columns of the window.
    Returns
    -------
    Window
        A new Window.
    Raises
    ------
    WindowError
        If a window can't be calculated.
    """
    if not isinstance(transform, rasterio.Affine):  # TODO: RPCs?
        raise WindowError("A transform object is required to calculate the window")

    rows, cols = rowcol(
            transform,
            [left, right, right, left],
            [top, top, bottom, bottom],
            op=float)

    row_start, row_stop = min(rows), max(rows)
    col_start, col_stop = min(cols), max(cols)

    return windows.Window.from_slices(
            (row_start, row_stop), (col_start, col_stop),
            width=max(col_stop - col_start, 0.0),
            height=max(row_stop - row_start, 0.0))

def reorient_bound(bound):
    def order(a, b):
        if b < a:
            return b, a
        return a, b

    l, b, r, t = bound
    if l < r and b < t:
        return bound
    else:
        l, r = order(l, r)
        b, t = order(b, t)
        return BoundingBox(l, b, r, t)

def intersect_bounds(bound1, bound2):
    bound1 = reorient_bound(bound1)
    bound2 = reorient_bound(bound2)

    int_l = max(bound1[0], bound2[0])
    int_b = max(bound1[1], bound2[1])
    int_r = min(bound1[2], bound2[2])
    int_t = min(bound1[3], bound2[3])

    if int_l < int_r and int_b < int_t:
        return BoundingBox(int_l, int_b, int_r, int_t)
    raise ValueError("disjoint bounds")

