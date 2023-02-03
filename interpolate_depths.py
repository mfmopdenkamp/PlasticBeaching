import pickle_manager as pickm
import pickle
import load_data
import pandas as pd
import numpy as np
from scipy.interpolate import griddata


ds = load_data.get_ds_drifters(filename='gdp_galapagos.nc')
df_shore = load_data.get_shoreline('c')
ds_gebco = load_data.get_bathymetry()
df_d2s = load_data.get_distance_to_shore_raster_04()


lon, lat = np.meshgrid(ds_gebco.lon, ds_gebco.lat)
drifter_depth = griddata((lon.flatten(), lat.flatten()), ds_gebco.elevation.flatten(), (ds.longitude, ds.latitude),
                         method='linear')

with open('drifter_depth.pkl', 'wb') as f:
    pickle.dump(drifter_depth, f)
