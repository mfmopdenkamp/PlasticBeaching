import pandas as pd
import numpy as np
import xarray as xr
from time import time
from file_names import *
from numba import jit

local = False
include_overlapping_months = False

df = pd.read_csv(f'{file_name_1}', parse_dates=['time'], infer_datetime_format=True)

# local test run: select october 2003 only
if local:
    df = df[(df.time >= '2003-10-01') & (df.time < '2003-11-01')]
else:
    # ERA5 data on lorenz is available from 1993 onwards only
    df = df[df.time >= '1993-01-01']

if not include_overlapping_months:
    time_plus_1h = df['time'] + pd.Timedelta('1h')
    mask = time_plus_1h.dt.month == df['time'].dt.month
    df = df[mask]


unique_months = df['time'].dt.strftime('%Y%m').unique().tolist()
df['hours_of_month'] = (df['time'].dt.day - 1) * 24 + df['time'].dt.hour
print('Total number of ERA5 files to load based on unique months:'
      f'num_unique_months = {len(unique_months)}')

#%%
@jit(nopython=True)
def get_wind_features(lats, lons, hours_of_month, u10, v10):
    n = len(lats)
    lat_grid = np.arange(90, -90.25, -0.25)
    lon_grid = np.arange(0, 360, 0.25)
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
    for i, (lat, lon, h_of_month) in enumerate(zip(lats, lons, hours_of_month)):
        i_lat = np.argmin(np.abs(lat_grid - lat))
        i_lon = np.argmin(np.abs(lon_grid - lon))
        u10_sel = u10[h_of_month:h_of_month+24, i_lat, i_lon]
        v10_sel = v10[h_of_month:h_of_month+24, i_lat, i_lon]
        abs_speed = np.hypot(u10_sel, v10_sel)
        abs_mean[i] = np.mean(abs_speed)
        u_mean[i] = np.mean(u10_sel)
        v_mean[i] = np.mean(v10_sel)
        abs_max[i] = np.max(abs_speed)
        u_max[i] = np.max(u10_sel)
        v_max[i] = np.max(v10_sel)
        abs_min[i] = np.min(abs_speed)
        u_min[i] = np.min(u10_sel)
        v_min[i] = np.min(v10_sel)
        abs_std[i] = np.std(abs_speed)
        u_std[i] = np.std(u10_sel)
        v_std[i] = np.std(v10_sel)

    return abs_mean, u_mean, v_mean, abs_max, u_max, v_max, abs_min, u_min, v_min, abs_std, u_std, v_std

if local:
    prefix_era5_data = 'data/reanalysis-era5-single-level_wind10m_'
else:
    prefix_era5_data = '/storage/shared/oceanparcels/input_data/ERA5/reanalysis-era5-single-level_wind10m_'

abs_mean = np.zeros(len(df))
u_mean = np.zeros(len(df))
v_mean = np.zeros(len(df))
abs_max = np.zeros(len(df))
u_max = np.zeros(len(df))
v_max = np.zeros(len(df))
abs_min = np.zeros(len(df))
u_min = np.zeros(len(df))
v_min = np.zeros(len(df))
abs_std = np.zeros(len(df))
u_std = np.zeros(len(df))
v_std = np.zeros(len(df))


start = time()
i_feature = 0
for YYYYMM in unique_months:
    df_sel = df[df['time'].dt.strftime('%Y%m') == YYYYMM]
    if include_overlapping_months:
        last_date = df_sel.time.iloc[-1]
        times = pd.date_range(last_date, last_date + pd.Timedelta(days=1), freq='H')
        YYYYMM_keys = times.strftime('%Y%m').unique().tolist()
        file_paths = [prefix_era5_data + yymm + '.nc' for yymm in YYYYMM_keys]
        if len(file_paths) > 1:
            ds = xr.open_mfdataset(file_paths)
        else:
            ds = xr.open_dataset(file_paths[0])
    else:
        ds = xr.open_dataset(prefix_era5_data + YYYYMM + '.nc')
    n = len(df_sel)
    abs_mean[i_feature:i_feature+n], u_mean[i_feature:i_feature+n], v_mean[i_feature:i_feature+n], \
    abs_max[i_feature:i_feature+n], u_max[i_feature:i_feature+n], v_max[i_feature:i_feature+n], \
    abs_min[i_feature:i_feature+n], u_min[i_feature:i_feature+n], v_min[i_feature:i_feature+n], \
    abs_std[i_feature:i_feature+n], u_std[i_feature:i_feature+n], v_std[i_feature:i_feature+n] = \
        get_wind_features(df_sel.latitude.values, df_sel.longitude.values, df_sel.hours_of_month.values, ds.u10.values, ds.v10.values)

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
