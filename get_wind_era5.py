
import pandas as pd
import numpy as np
import xarray as xr

df = pd.read_csv('data/events.csv', parse_dates=['time_start', 'time_end'], infer_datetime_format=True)

prefix_era5_data = 'data/reanalysis-era5-single-level_wind10m_'
n = df.shape[0]
V_mean = np.zeros(n)
u_mean = np.zeros(n)
v_mean = np.zeros(n)
V_max = np.zeros(n)
u_max = np.zeros(n)
v_max = np.zeros(n)
V_std = np.zeros(n)
u_std = np.zeros(n)
v_std = np.zeros(n)


for i_event, event in enumerate(df.itertuples()):
    print(f'Do something from start time : {event.time_start} until : {event.time_end}')
    times = pd.date_range(event.time_start, event.time_end, freq='H')
    YYYYMM_keys = np.unique([f'{year}{("0" if month < 10 else "")}{month}' for year, month in zip(times.year, times.month)])
    dss = []
    for yyyymm in YYYYMM_keys:
        dss.append(xr.open_dataset(prefix_era5_data+yyyymm))
        print('Open '+prefix_era5_data+yyyymm)
    ds = xr.concat(dss, dim='time')

    # interpolate lons and lats
    lat = ds.latidude[np.argmin(abs(ds.latitude - event.latitude))]
    lon = ds.longitude[np.argmin(abs(ds.longitude - event.longitude))]
    ds = ds.sel(time=times, latitude=lat, longitude=lon)
    abs_speed = np.hypot(ds.u, ds.v)
    V_mean[i_event] = np.mean(abs_speed)
    u_mean[i_event] = np.mean(ds.u)
    v_mean[i_event] = np.mean(ds.v)
    V_max[i_event] = np.max(abs_speed)
    u_max[i_event] = np.max(ds.u)
    v_max[i_event] = np.max(ds.v)
    V_std[i_event] = np.std(abs_speed)
    u_std[i_event] = np.std(ds.u)
    v_std[i_event] = np.std(ds.v)

df.assign(V_mean=V_mean,
          u_mean=u_mean,
          v_mean=v_mean,
          V_max=V_max,
          u_max=u_max,
          v_max=v_max,
          V_std=V_std,
          u_std=u_std,
          v_std=v_std
          )
