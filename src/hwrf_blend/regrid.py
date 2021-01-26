import eccodes
import ESMF
import xarray as xr
import numpy as np

class GribMessage:
    __slots__ = ['msgid', 'closed']

    def __init__(self, msgid):
        self.closed = False
        self.msgid = msgid

    def __del__(self):
        self.close()

    def close(self):
        if not self.closed:
            eccodes.codes_release(self.msgid)
            self.closed = True

    def __hash__(self):
        return self.msgid

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.msgid == other.msgid

    def __getitem__(self, key):
        if eccodes.codes_is_defined(self.msgid, key):
            if eccodes.codes_get_size(self.msgid, key) > 1:
                return eccodes.codes_get_array(self.msgid, key)
            else:
                return eccodes.codes_get(self.msgid, key)
        raise KeyError(f"{key} does not exist")

    def _key_iterator(self):
        if self.closed:
            return

        iterid = eccodes.codes_keys_iterator_new(self.msgid)
        while eccodes.codes_keys_iterator_next(iterid):
            k = eccodes.codes_keys_iterator_get_name(iterid)
            if eccodes.codes_is_defined(self.msgid, k):
                yield k
        eccodes.codes_keys_iterator_delete(iterid)

    def keys(self):
        return list(self._key_iterator())

    def values(self):
        rv = []

        for k in self._key_iterator():
            rv.append(self[k])
        return rv

    def items(self):
        rv = {}

        for k in self._key_iterator():
            rv[k] = self[k]
        return rv

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

def get_grib_latlon(filename):
    with open(filename, 'rb') as fh:
        f = eccodes.codes_grib_new_from_file(fh)
        x = eccodes.codes_get(f, 'Ni')
        y = eccodes.codes_get(f, 'Nj')
        arrla = eccodes.codes_get_double_array(f, 'latitudes')
        arrlo = eccodes.codes_get_double_array(f, 'longitudes')
        eccodes.codes_release(f)
    arrla = arrla.reshape((x, y))
    arrlo = arrlo.reshape((x, y))
    return arrla, arrlo

def get_2t(filename):
    with open(filename, 'rb') as fh:
        f = eccodes.codes_grib_new_from_file(fh)
        x = eccodes.codes_get(f, 'Ni')
        y = eccodes.codes_get(f, 'Nj')
        iterid = eccodes.codes_get_iterator_new(f)
        while True:
            result = eccodes.codes_grib_iterator_next(iterid)
            key_name
            lat, lon, value = result
        arr2t = eccodes.codes_get_double_array(f, '2t')

def get_nwm_grid(filename):
    with xr.open_dataset(filename) as ds:
        ds = ds.sel({"Time": 0})
        xlat = ds.XLAT_M.values
        xlong = ds.XLONG_M.values
    return xlat, xlong


