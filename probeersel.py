import pickle_manager as pickm
import load_data
import pandas as pd
import numpy as np
from interpolate_drifter_location import interpolate_drifter_location

filename = 'gdp_v2.00.nc'
ds = pickm.load_pickle_wrapper(filename, load_data.drifter_data_hourly, filename)

filename = 'dist2coast.txt.bz2'
df_shore = pickm.load_pickle_wrapper(filename, pd.read_csv, load_data.data_folder+filename,
                                     delim_whitespace=True, names=['longitude', 'latitude', 'distance'],
                                     header=None, compression='bz2')

drifter_dist_approx = pickm.load_pickle_wrapper('drifter_distances_interpolated_0.04deg_raster',
                                                interpolate_drifter_location, df_shore, ds)

proximity = 10  # kilometers
close_to_shore = drifter_dist_approx < proximity
print(f'Total rows left = {close_to_shore.sum()}')
ds_subset = ds.isel(obs=np.where(close_to_shore)[0])
