import numpy as np
import picklemanager as pickm
import pandas as pd
import toolbox as tb
import load_data
import plotter
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


# %% Settings

percentage = 100
random_set = 1
gps_only = True
undrogued_only = True
threshold_aprox_distance_km = 12
start_date = '2015-11-22'
end_date = '2015-11-26'

name = f'subset_{percentage}{(f"_{random_set}" if percentage < 100 else "")}'\
       f'{("_" + str(start_date) if start_date is not None else "")}' \
       f'{("_" + str(end_date) if end_date is not None else "")}' \
       f'{("_gps" if gps_only else "")}' \
       f'{("_undrogued" if undrogued_only else "")}' \
       f'{("_" + str(threshold_aprox_distance_km) + "km" if threshold_aprox_distance_km is not None else "")}'

ds_gdp = pickm.pickle_wrapper('ds_gdp_' + name, load_data.load_subset, percentage, gps_only, undrogued_only,
                              threshold_aprox_distance_km, start_date, end_date)

obs = ds_gdp.obs.values[ds_gdp.ids == 300234061407950]
traj = tb.traj_from_obs(ds_gdp, obs)
ds_gdp = ds_gdp.sel(obs=obs, traj=traj)

min_lon, max_lon, min_lat, max_lat = ds_gdp.longitude.min(), ds_gdp.longitude.max(), ds_gdp.latitude.min(), ds_gdp.latitude.max()
delta_lon = max_lon - min_lon
delta_lat = max_lat - min_lat

fig, ax = plotter.get_marc_subplots(extent=[min_lon - 0.2 * delta_lon, max_lon + 0.2 * delta_lon,
                                            min_lat - 0.2 * delta_lat, max_lat + 0.2 * delta_lat])

ax.scatter(ds_gdp.longitude, ds_gdp.latitude, transform=ccrs.PlateCarree())

ax.set_ylabel('Latitude N')
ax.set_xlabel('Longitude E')
plt.tight_layout()
plt.show()


