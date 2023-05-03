
import pandas as pd
import numpy as np
import xarray as xr

name = 'subtrajs_random_subset_5_2_gps_only_undrogued_only'

df = pd.read_csv(f'data/{name}.csv',
                 parse_dates=['time_start', 'time_end'], index_col='ID',
                 infer_datetime_format=True)

prefix_era5_data = '/storage/shared/oceanparcels/input_data/ERA5/reanalysis-era5-single-level_wind10m_'
n = df.shape[0]
V_mean = np.zeros(n)
u_mean = np.zeros(n)
v_mean = np.zeros(n)
V_max = np.zeros(n)
u_max = np.zeros(n)
v_max = np.zeros(n)
V_min = np.zeros(n)
u_min = np.zeros(n)
v_min = np.zeros(n)
V_std = np.zeros(n)
u_std = np.zeros(n)
v_std = np.zeros(n)


for i_event, subtraj in enumerate(df.itertuples()):
    print(f'Do something from start time : {subtraj.time_start} until : {subtraj.time_end}')
    times = pd.date_range(subtraj.time_start, subtraj.time_end, freq='H')
    YYYYMM_keys = np.unique([f'{year}{("0" if month < 10 else "")}{month}' for year, month in zip(times.year, times.month)])
    file_paths = [prefix_era5_data+yymm+'.nc' for yymm in YYYYMM_keys]
    if len(file_paths) > 1:
        ds = xr.open_mfdataset(file_paths)
    else:
        ds = xr.open_dataset(file_paths[0])

    # interpolate lons and lats
    lat = ds.latitude[np.argmin(abs(ds.latitude.values - subtraj.latitude_start))]
    lon = ds.longitude[np.argmin(abs(ds.longitude.values - subtraj.longitude_start))]
    ds = ds.sel(time=times, latitude=lat, longitude=lon)
    abs_speed = np.hypot(ds.u10, ds.v10)
    V_mean[i_event] = np.mean(abs_speed)
    u_mean[i_event] = np.mean(ds.u10)
    v_mean[i_event] = np.mean(ds.v10)
    V_max[i_event] = np.max(abs_speed)
    u_max[i_event] = np.max(ds.u10)
    v_max[i_event] = np.max(ds.v10)
    V_min[i_event] = np.min(abs_speed)
    u_min[i_event] = np.min(ds.u10)
    v_min[i_event] = np.min(ds.v10)
    V_std[i_event] = np.std(abs_speed)
    u_std[i_event] = np.std(ds.u10)
    v_std[i_event] = np.std(ds.v10)

    ds.close()

df = df.assign(V_mean=V_mean,
               u_mean=u_mean,
               v_mean=v_mean,
               V_max=V_max,
               u_max=u_max,
               v_max=v_max,
               V_min=V_min,
               u_min=u_min,
               v_min=v_min,
               V_std=V_std,
               u_std=u_std,
               v_std=v_std
               )

df.to_csv(f'data/{name}_wind.csv')
