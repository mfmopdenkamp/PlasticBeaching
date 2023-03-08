# This script slightly reduces the size of the gdp hourly dataset by removing the data variables
# related to SST.

import load_data
import xarray as xr
import picklemanager as pickm

ds = load_data.get_ds_drifters('gdp_v2.00.nc_approx_dist_nearest')

ds['aprox_distance_shoreline'] = xr.DataArray(
        data=ds['aprox_distance_shoreline'],
        dims='obs',
        attrs={'long_name': 'Approximate distance to shoreline by interpolation onto 0.04deg raster',
               'units': 'km'})


def drop_sst(ds):
    return ds.drop_vars(['sst', 'sst2', 'err_sst', 'err_sst2', 'flg_sst', 'flg_sst1','flg_sst2'])


pickm.pickle_wrapper('gdp_v2.00.nc_no_sst', drop_sst, ds)

