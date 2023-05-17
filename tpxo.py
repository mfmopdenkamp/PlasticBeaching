import xarray as xr
import pandas as pd
import numpy as np
import config
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs


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


ds_m2 = xr.load_dataset('data/h_m2_tpxo9_atlas_30.nc')

# Create a meshgrid of longitudes and latitudes
lons = ds_m2.lon_z.values
lats = ds_m2.lat_z.values

# Create the figure and axis with a specific projection
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Add land and ocean background
ax.add_feature(cartopy.feature.LAND, facecolor='lightgray')
ax.add_feature(cartopy.feature.OCEAN, facecolor='white')

# Plot the data as a colormap using imshow
cmap = plt.cm.get_cmap('jet')
im = ax.imshow(ds_m2.hRe.values, extent=(lons.min(), lons.max(), lats.min(), lats.max()),
               origin='lower', cmap=cmap)

# Add a colorbar
cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7)
cbar.set_label('Data values')

# Customize the plot
ax.set_title('Colormap on Oceans')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Show coastlines
ax.coastlines()

# Show the plot
plt.show()
