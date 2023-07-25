import pandas as pd
import numpy as np
import xarray as xr
from time import time
from file_names import *

local = False

df = pd.read_csv(f'{file_name_1}', parse_dates=['time'], infer_datetime_format=True)

# local test run: select october 2003 only
if local:
    df = df[(df.time >= '2003-10-01') & (df.time < '2003-10-30')]
else:
    # ERA5 data on lorenz is available from 1993 onwards only
    df = df[df.time >= '1993-01-01']



print('Total number of ERA5 files to load based on unique months:'
      f'num_unique_months = {df.time.dt.to_period("M").nunique()}')

#%%
n = df.shape[0]
abs_mean = np.zeros(n)
u_mean = np.zeros(n)
v_mean = np.zeros(n)
abs_max = np.zeros(n)
u_max = np.zeros(n)
v_max = np.zeros(n)
abs_min = np.zeros(n)
u_min = np.zeros(n)
v_min = np.zeros(n)
abs_std = np.zeros(n)
u_std = np.zeros(n)
v_std = np.zeros(n)

if local:
    prefix_era5_data = 'data/reanalysis-era5-single-level_wind10m_'
else:
    prefix_era5_data = '/storage/shared/oceanparcels/input_data/ERA5/reanalysis-era5-single-level_wind10m_'

prev_file_paths = []
start = time()
for i_event, drifter_state in enumerate(df.itertuples()):
    times = pd.date_range(drifter_state.time, drifter_state.time + pd.Timedelta(days=1), freq='H')
    YYYYMM_keys = np.unique([f'{year}{("0" if month < 10 else "")}{month}' for year, month in zip(times.year, times.month)])
    file_paths = [prefix_era5_data+yymm+'.nc' for yymm in YYYYMM_keys]
    if file_paths != prev_file_paths:
        if len(file_paths) > 1:
            ds = xr.open_mfdataset(file_paths)
        else:
            ds = xr.open_dataset(file_paths[0])

    # interpolate lons and lats
    lat = ds.latitude[np.argmin(abs(ds.latitude.values - drifter_state.latitude))]
    lon = ds.longitude[np.argmin(abs(ds.longitude.values - drifter_state.longitude))]
    ds = ds.sel(time=times, latitude=lat, longitude=lon)
    abs_speed = np.hypot(ds.u10, ds.v10)
    abs_mean[i_event] = np.mean(abs_speed)
    u_mean[i_event] = np.mean(ds.u10)
    v_mean[i_event] = np.mean(ds.v10)
    abs_max[i_event] = np.max(abs_speed)
    u_max[i_event] = np.max(ds.u10)
    v_max[i_event] = np.max(ds.v10)
    abs_min[i_event] = np.min(abs_speed)
    u_min[i_event] = np.min(ds.u10)
    v_min[i_event] = np.min(ds.v10)
    abs_std[i_event] = np.std(abs_speed)
    u_std[i_event] = np.std(ds.u10)
    v_std[i_event] = np.std(ds.v10)

    file_path_prev = file_paths

total_time = time() - start
print(f'Total time to run for loop loading ERA5 data: {total_time:.2f} seconds')

df = df.assign(wind10m_abs_mean=abs_mean,
               wind10m_u_mean=u_mean,
               wind10m_v_mean=v_mean,
               wind10m_abs_max=abs_max,
               wind10m_u_max=u_max,
               wind10m_v_max=v_max,
               wind10m_abs_min=abs_min,
               wind10m_u_min=u_min,
               wind10m_v_min=v_min,
               wind10m_abs_std=abs_std,
               wind10m_u_std=u_std,
               wind10m_v_std=v_std
               )

df.to_csv(f'{file_name_2}.csv', index=False)
