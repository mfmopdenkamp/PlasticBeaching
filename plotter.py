import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd

import load_data
import numpy as np
from tqdm import tqdm


death_type_colors = {0: 'r', 1: 'g', 2: 'b', 3: 'y', 4: 'c', 5: 'm', 6: 'aquamarine'}


def get_sophie_subplots(figsize=(12, 8), extent=(-92.5, -88.5, -1.75, 0.75), title=''):
    """ This function sets up a figure (fig and ax) for plotting the data in the Galapagos region.
    This set-up contains a background terrain map of the Galapagos region, extending from lat (-2,1) and lon (-93, -88.5)"""
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    from cartopy.io.img_tiles import Stamen

    fig = plt.figure(figsize=figsize, dpi=300)

    tiler = Stamen('terrain-background')
    mercator = tiler.crs
    ax = plt.axes(projection=mercator)

    ax.set_extent(extent, crs=ccrs.Geodetic())

    zoom = 10  # trial and error number, too big, and things don't load, too small, and map is low resolution
    ax.add_image(tiler, zoom)

    plt.title(title, fontsize=20)

    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    return fig, ax


def get_marc_subplots(size=(12, 8), extent=(-92.5, -88.5, -1.75, 0.75), title=''):
    """ This function sets up a figure (fig and ax) for plotting the data in lonlatbox of the extent."""

    fig = plt.figure(figsize=size, dpi=300)

    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent(extent, crs=ccrs.Geodetic())

    plt.title(title, fontsize=20)

    # gl = ax.gridlines(draw_labels=True)
    # gl.top_labels = gl.right_labels = False

    return fig, ax


def plot_trajectories_death_type(ds, s=2):
    """given a dataset, plot the trajectories on a map"""
    plt.figure(figsize=(12, 8), dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    for death_type in np.unique(ds.type_death):
        traj = np.where(ds.type_death == death_type)[0]
        ds_traj = ds.isel(obs=np.where(np.isin(ds.ids, ds.ID[traj]))[0], traj=traj)
        lat = ds_traj.latitude
        lon = ds_traj.longitude
        ax.scatter(lon, lat, transform=ccrs.PlateCarree(), s=s, c=death_type_colors[death_type])

    ax.coastlines()
    ax.set_ylabel('Latitude N')
    ax.set_xlabel('Longitude E')
    plt.tight_layout()
    plt.show()


def plot_beaching_trajectories(ds, ax=None, s=15, ds_beaching_obs=None, df_shore=pd.DataFrame()):
    """given a dataset, plot the trajectories on a map"""
    if ax is None:
        plt.figure(figsize=(12, 8), dpi=300)
        ax = plt.axes(projection=ccrs.PlateCarree())
        extent_offset = 0.2
        ax.set_xlim([ds['longitude'].min()-extent_offset, ds['longitude'].max()+extent_offset])
        ax.set_ylim([ds['latitude'].min()-extent_offset, ds['latitude'].max()+extent_offset])

    ax.scatter(ds.longitude, ds.latitude, transform=ccrs.PlateCarree(), s=s, c='midnightblue')
    if ds_beaching_obs is not None:
        ax.scatter(ds_beaching_obs.longitude, ds_beaching_obs.latitude, s=s, c='r')
    ax.plot(ds.longitude, ds.latitude, ':k', transform=ccrs.PlateCarree(), alpha=0.5)

    if not df_shore.empty:
        df_shore.plot(ax=ax, color='b')
    # else:
    #     ax.coastlines()

    plt.tight_layout()
    plt.show()


def plot_galapagos_map_distances(ds, title=''):
    fig, ax = get_sophie_subplots(extent=(-92, -88.5, -1.75, 1), title=title)
    pcm = ax.scatter(ds.longitude, ds.latitude, transform=ccrs.PlateCarree(),
                     c=ds.distance_shoreline, cmap='inferno')
    ax.scatter(ds.longitude, ds.latitude, transform=ccrs.PlateCarree(), c='k', s=0.4,
               label='all', alpha=0.8)
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label('Distance to shore')
    ax.set_ylabel('Latitude N')
    ax.set_xlabel('Longitude E')
    plt.show(bbox_inches='tight')


def plot_last_distances(ds, last_hours=100, max_drifters=100):
    ids = np.unique(ds.ids)

    fig, ax = plt.subplots()
    for i, ID in enumerate(tqdm(ids)):
        ds_id = ds.isel(obs=np.where(ds.ids == ID)[0])
        distance = ds_id.distance_shoreline.values[:-last_hours:-1]
        plt.plot(np.arange(len(distance)), distance/1000, label=str(ID))
        if i+1 == max_drifters:
            break

    ax.set_xlabel('hours to last data point')
    ax.set_ylabel('distance to the shoreline [km]')
    plt.title(f'Last {last_hours} hours of {i+1} drifters')

    plt.show()


def plot_velocity_hist(ds):
    fig, ax = plt.subplots()
    ax.hist(np.hypot(ds.ve, ds.vn), bins=50)
    ax.set_xlabel('velocity [m/s]')
    ax.set_ylabel('# data points')
    plt.show()


def plot_distance_hist(ds):
    fig, ax = plt.subplots()
    ax.hist(ds.distance_shoreline, bins=50)
    ax.set_xlabel('distance to the shoreline [m]')
    ax.set_ylabel('# data points')
    plt.show()


def plot_death_type_bar(ds):
    death_types, n_death_types = np.unique(ds.type_death, return_counts=True)

    fig, ax = plt.subplots()
    ax.bar(death_types, n_death_types, color=death_type_colors.values())
    ax.set_xlabel('death type')
    ax.set_ylabel('# drifters')
    plt.xticks(death_types)
    plt.show()


def plot_velocity_distance(ds):
    fig, ax = plt.subplots()
    ax.scatter(ds.distance_shoreline, np.hypot(ds.ve, ds.vn))
    ax.set_ylabel('velocity [m/s]')
    ax.set_xlabel('distance to the shoreline [m]')
    plt.semilogy()
    plt.show()


def plot_uniques_bar(ds_array, xlabel):
    unique_values, counts = np.unique(ds_array, return_counts=True)
    n = len(unique_values)

    fig, ax = plt.subplots(figsize=(min(18, max(n, 8)), 8))
    ax.bar(np.arange(n), counts)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('# drifters')
    plt.xticks(np.arange(n))
    ax.set_xticklabels(unique_values, rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    gdp = load_data.drifter_data_six_hourly(30000)

    ax = plt.axes(projection=ccrs.PlateCarree())
    for ID in gdp['ID'].unique():
        ax.plot(gdp[gdp['ID'] == ID]['Longitude'], gdp[gdp['ID'] == ID]['Latitude'], transform=ccrs.PlateCarree())
        break

    ax.coastlines()
    plt.show()


