import load_data
import picklemanager as pickm
from analyzer import interpolate_drifter_location
import xarray as xr


def add_interpol_nearest():
    ds = load_data.get_ds_drifters(filename='gdp_v2.00.nc')
    df_raster = load_data.get_raster_distance_to_shore_04deg()

    ds['aprox_distance_shoreline'] = xr.DataArray(
        data=interpolate_drifter_location(df_raster, ds, method='linear'),
        dims='obs',
        attrs={'long_name': 'Approximate distance to shoreline by interpolation onto 0.04deg raster',
               'units': 'km'})

    return ds


pickle_name = 'gdp_v2.00.nc_approx_dist_linear'
ds = pickm.pickle_wrapper(pickle_name, add_interpol_nearest)
