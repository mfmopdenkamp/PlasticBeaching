import xarray as xr
import pandas as pd
import numpy as np
import toolbox as a
import matplotlib
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import os

plot_test = True
basic_file = 'subtrajs_random_subset_5_2_gps_only_undrogued_only'
wind_file = basic_file + '_wind2.csv'
tide_file = basic_file + '_wind2_tides.csv'

if os.path.exists('data/tidals/' + tide_file):
    df_segments = pd.read_csv('data/tidals' + tide_file)
else:

    df_segments = pd.read_csv('data/'+wind_file, parse_dates=['time_start', 'time_end'])

    for tidal_contituent in ['m2', 'm4', 's2', 'n2', 'k1', 'k2', 'o1', 'p1', 'q1']:
        ds = xr.open_dataset(f'data/h_{tidal_contituent}_tpxo9_atlas_30.nc')

        new_column = np.zeros(df_segments.shape[0], dtype=int)
        lats = df_segments.latitude_start.values
        lons = a.longitude_translator(df_segments.longitude_start.values.copy())
        for i, (lat, lon) in enumerate(zip(lats, lons)):

            i_lat = np.argmin(abs(ds.lat_z.values - lat))
            i_lon = np.argmin(abs(ds.lon_z.values - lon))

            new_column[i] = ds.hRe.values[i_lon, i_lat]


        df_segments[f'{tidal_contituent}_tidal_elevation_mm'] = new_column

        ds.close()

    df_segments.to_csv('data/'+tide_file, index=False)


if plot_test:
    ds_m2 = xr.load_dataset('data/h_m2_tpxo9_atlas_30.nc')
    plt.ioff()
    # matplotlib.use('TkAgg')
    def plot_m2_tidal_elevation(lon, lat, time, m2_elevation):
        # Select the area around a segment start
        min_lon, max_lon, min_lat, max_lat = a.get_lonlatbox(lon, lat, 100000)
        min_lon2 = a.longitude_translator(min_lon)
        max_lon2 = a.longitude_translator(max_lon)
        nx_min = np.argmin(abs(ds_m2.lon_z.values - min_lon2))
        nx_max = np.argmin(abs(ds_m2.lon_z.values - max_lon2))
        ny_min = np.argmin(abs(ds_m2.lat_z.values - min_lat))
        ny_max = np.argmin(abs(ds_m2.lat_z.values - max_lat))
        ds_m2_sel = ds_m2.isel(nx=slice(nx_min, nx_max), ny=slice(ny_min, ny_max))

        # Create the figure and axis with a specific projection

        fig = plt.figure()
        ax = fig.add_subplot(projection=ccrs.PlateCarree())
        # Add land and ocean background
        ax.add_feature(cartopy.feature.LAND, facecolor='lightgray')
        ax.add_feature(cartopy.feature.OCEAN, facecolor='white')

        # Plot the data as a colormap using imshow
        im = ax.imshow(ds_m2_sel.hRe.values.T, extent=[min_lon, max_lon, min_lat, max_lat],
                       origin='lower', cmap='jet')

        # Add a colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7)
        cbar.set_label('Tidal elevation (mm)')

        # Customize the plot
        ax.set_title('M2 tidal constituent')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Turn on the grid
        ax.grid(True, linestyle='--', color='gray')

        # Show xticks and yticks at 0.5 intervals on a 0.1 degree grid
        ax.set_xticks(np.arange(min_lon, max_lon, 0.5))
        ax.set_yticks(np.arange(min_lat, max_lat, 0.5))

        # Plot coordinates and annotate the elevation with arrow
        ax.plot(lon, lat, 'ko')
        ax.annotate(f'{m2_elevation} mm', xy=(lon, lat), xytext=(lon + 0.1, lat + 0.1))

        # Display the date above the colorbar
        ax.text(1.02, 0.9, time, transform=ax.transAxes)

        # Show coastlines
        ax.coastlines()

        # Show the plot
        plt.show()


    for i in df_segments.index[df_segments['beaching_flag']][35:37]:
        plot_m2_tidal_elevation(df_segments['longitude_start'].values[i], df_segments['latitude_start'].values[i],
                                df_segments['time_start'].values[i],
                                df_segments['m2_tidal_elevation_mm'].values[i])
