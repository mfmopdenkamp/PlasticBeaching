import pickle_manager as pickm
import load_data

filename = 'gdp_v2.00.nc'
ds = pickm.load_pickle_wrapper(filename, load_data.drifter_data_hourly, filename)

print(ds.dims)

print(ds.data_vars)

print(ds.attrs)
