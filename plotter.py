import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import load_data
import numpy as np
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.img_tiles import Stamen
from tqdm import tqdm


def get_sophie_subplots(size=(12, 8), extent=(-92.5, -88.5, -1.75, 0.75), title=''):
    """ This function sets up a figure (fig and ax) for plotting the data in the Galapagos region.
    This set-up contains a background terrain map of the Galapagos region, extending from lat (-2,1) and lon (-93, -88.5)"""
    fig = plt.figure(figsize=size, dpi=300)

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


def plot_trajectories(ds):
    """given a dataset, plot the trajectories on a map"""


    pass


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


if __name__ == '__main__':
    gdp = load_data.drifter_data_six_hourly(30000)

    ax = plt.axes(projection=ccrs.PlateCarree())
    for ID in gdp['ID'].unique():
        ax.plot(gdp[gdp['ID'] == ID]['Longitude'], gdp[gdp['ID'] == ID]['Latitude'], transform=ccrs.PlateCarree())
        break

    ax.coastlines()
    plt.show()


