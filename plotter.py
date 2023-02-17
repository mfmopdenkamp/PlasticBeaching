import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import load_data
from matplotlib import colors, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.img_tiles import OSM
import cartopy.feature as cfeature
from cartopy.io import shapereader
from cartopy.io.img_tiles import Stamen
from cartopy.io.img_tiles import GoogleTiles
from matplotlib.transforms import offset_copy


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


if __name__ == '__main__':
    gdp = load_data.drifter_data_six_hourly(30000)

    ax = plt.axes(projection=ccrs.PlateCarree())
    for ID in gdp['ID'].unique():
        ax.plot(gdp[gdp['ID'] == ID]['Longitude'], gdp[gdp['ID'] == ID]['Latitude'], transform=ccrs.PlateCarree())
        break

    ax.coastlines()
    plt.show()


