import xarray as xr
import pandas as pd
import numpy as np
import analyzer as a
import config
import matplotlib
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import os

if os.path.exists('data/'+config.csv_wind2_tides):
    df_segments = pd.read_csv('data/'+config.csv_wind2_tides)
else:

    df_segments = pd.read_csv('data/'+config.csv_wind2)

    for tidal_contituent in ['m2', 'm4', 's2', 'n2', 'k1', 'k2', 'o1', 'p1', 'q1']:
        ds = xr.open_dataset(f'data/h_{tidal_contituent}_tpxo9_atlas_30.nc')

        new_column = np.zeros(df_segments.shape[0], dtype=int)

        for i, (lat, lon) in enumerate(zip(df_segments.latitude_start.values, df_segments.longitude_start.values)):
            i_lat = np.argmin(abs(ds.lat_z.values - lat))
            i_lon = np.argmin(abs(ds.lon_z.values - lon))

            new_column[i] = ds.hRe.values[i_lon, i_lat]

        df_segments[f'{tidal_contituent}_tidal_elevation_mm'] = new_column

        ds.close()

    df_segments.to_csv('data/'+config.csv_wind2_tides, index=False)


ds_m2 = xr.load_dataset('data/h_m2_tpxo9_atlas_30.nc')
# convert lon_z and lat_z variables to coordinates
ds_m2 = ds_m2.assign_coords(lat=ds_m2.lat_z, lon=ds_m2.lon_z)

# Select the area around a segment start
min_lon, max_lon, min_lat, max_lat = a.get_lonlatbox(df_segments['longitude_start'][0], df_segments['latitude_start'][0], 500000)
nx_min = np.argmin(abs(ds_m2.lon_z.values - min_lon))
nx_max = np.argmin(abs(ds_m2.lon_z.values - max_lon))
ny_min = np.argmin(abs(ds_m2.lat_z.values - min_lat))
ny_max = np.argmin(abs(ds_m2.lat_z.values - max_lat))
ds_m2_sel = ds_m2.isel(nx=slice(nx_min, nx_max), ny=slice(ny_min, ny_max))


# Create the figure and axis with a specific projection
plt.ioff()
matplotlib.use('TkAgg')
fig = plt.figure()
ax = fig.add_subplot(projection=ccrs.PlateCarree())
# Add land and ocean background
ax.add_feature(cartopy.feature.LAND, facecolor='lightgray')
ax.add_feature(cartopy.feature.OCEAN, facecolor='white')

# Plot the data as a colormap using imshow

im = ax.imshow(ds_m2_sel.hRe.values.T, extent=[ds_m2_sel.lon_z.min(), ds_m2_sel.lon_z.max(), ds_m2_sel.lat_z.min(), ds_m2_sel.lat_z.max()],
               origin='lower', cmap='jet')

# Add a colorbar
cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7)
cbar.set_label('Tidal elevation (m)')

# Customize the plot
ax.set_title('M2 tidal constituent')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Show coastlines
ax.coastlines()

# Show the plot
plt.show()
