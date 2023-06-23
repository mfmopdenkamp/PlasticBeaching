import load_data
import picklemanager as pickm
import xarray as xr
import time
from scipy.interpolate import griddata
import numpy as np


def interpolate_drifter_location(df_raster, ds_drifter, method='linear'):
    # Drifter locations (longitude and latitude)
    drifter_lon = ds_drifter.longitude.values
    drifter_lat = ds_drifter.latitude.values

    # Shore distances (longitude, latitude and distance to the shore)
    raster_lon = df_raster.longitude.values
    raster_lat = df_raster.latitude.values
    raster_dist = df_raster.distance.values

    # Interpolate the drifter locations onto the raster
    start = time.time()
    print('Started interpolation...', end='')
    drifter_dist = griddata((raster_lon, raster_lat), raster_dist, (drifter_lon, drifter_lat), method=method)
    print(f'Done. Elapsed time {np.round(time.time() - start, 2)}s')

    return drifter_dist


def add_interpol(ds, method='nearest'):
    df_raster = load_data.get_raster_distance_to_shore_04deg()

    ds['aprox_distance_shoreline'] = xr.DataArray(
        data=interpolate_drifter_location(df_raster, ds, method=method),
        dims='obs',
        attrs={'long_name': 'Approximate distance to shoreline by interpolation onto 0.04deg raster',
               'units': 'km'})
    return ds


ds = load_data.get_ds_drifters(filename='gdp_v2.00.nc')
method = 'linear'
pickle_name = 'gdp_v2.00.nc_approx_dist_' + method
ds = pickm.pickle_wrapper(pickle_name, add_interpol, ds, method=method)
