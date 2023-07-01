import numpy as np
import load_data
import toolbox as tb
import xarray as xr
import picklemanager as pickm

ds = load_data.load_subset()

drogue_presence, lats, lons = tb.get_drogue_presence(ds, coords=True)

ds['drogue_presence'] = xr.DataArray(
        data=drogue_presence,
        dims='obs',
        attrs={'long_name': 'Boolean indication the presence of a drogue',
               'units': '-'})

# create latitude and longitude variables when drogue was lost


ds['latitude_drogue_lost'] = xr.DataArray(
    data=lats,
    dims='traj',
    attrs={'long_name': 'Latitude when drogue was lost',
              'units': 'degrees_north'})

ds['longitude_drogue_lost'] = xr.DataArray(
    data=lons,
    dims='traj',
    attrs={'long_name': 'Longitude when drogue was lost',
                'units': 'degrees_east'})

pickm.dump_pickle(ds, base_name='gdp_drogue_presence')