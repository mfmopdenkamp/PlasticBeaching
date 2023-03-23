
import pandas as pd
import numpy as np
import xarray as xr

df = pd.read_csv('data/events.csv', parse_dates=['time_start', 'time_end'], infer_datetime_format=True)

prefix_era5_data = '/storage/shared/oceanparcels/input_data/ERA5/reanalysis-era5-single-level_wind10m_'
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
        dss.append(xr.open_dataset(f'{prefix_era5_data}{yyyymm}.nc'))
        print('Open '+prefix_era5_data+yyyymm)
    ds = xr.concat(dss, dim='time')

    # interpolate lons and lats
    lat = ds.latitude[np.argmin(abs(ds.latitude - event.latitude_start))]
    lon = ds.longitude[np.argmin(abs(ds.longitude - event.longitude_start))]
    ds = ds.sel(time=times, latitude=lat, longitude=lon)
    abs_speed = np.hypot(ds.u10, ds.v10)
    V_mean[i_event] = np.mean(abs_speed)
    u_mean[i_event] = np.mean(ds.u10)
    v_mean[i_event] = np.mean(ds.v10)
    V_max[i_event] = np.max(abs_speed)
    u_max[i_event] = np.max(ds.u10)
    v_max[i_event] = np.max(ds.v10)
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
               V_std=V_std,
               u_std=u_std,
               v_std=v_std
               )

df.to_csv('data/events_wind.csv')
