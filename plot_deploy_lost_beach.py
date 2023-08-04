import load_data
import picklemanager
import picklemanager as pickm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import toolbox as tb
import plotter

percentage = 100
ylim = [-88, 88]
xlim = None
latlon_box_size = 2

type_death = 1
pickle_name = pickm.create_pickle_ds_gdp_name(type_death=type_death)
ds = pickm.pickle_wrapper(pickle_name, load_data.load_subset, type_death=type_death, ds_name='gdp_drogue_presence')


# %% Plot total
pickle_name = pickm.create_pickle_path(f'density_deploy_{percentage}_{latlon_box_size}_type_death_{type_death}')
try:
    (X, Y, density_grid_deploy) = picklemanager.load_pickle(pickle_name)
except FileNotFoundError:

    X, Y, density_grid_deploy = tb.get_density_grid(ds.deploy_lat.values, ds.deploy_lon.values, ylim=ylim, xlim=xlim,
                                             latlon_box_size=latlon_box_size)

    pickm.dump_pickle((X, Y, density_grid_deploy), pickle_name)

pickle_name = pickm.create_pickle_path(f'density_lost_drogue_{percentage}_{latlon_box_size}_type_death_{type_death}')
try:
    (X, Y, density_grid_lost) = pickm.load_pickle(pickle_name)
except FileNotFoundError:
    X, Y, density_grid_lost = tb.get_density_grid(ds.latitude_drogue_lost, ds.longitude_drogue_lost, xlim=xlim,
                                             ylim=ylim, latlon_box_size=latlon_box_size)
    pickm.dump_pickle((X, Y, density_grid_lost), pickle_name)

pickle_name_undrogued = pickm.create_pickle_path(f'density_end_{percentage}_{latlon_box_size}_type_death_{type_death}')
try:
    (X, Y, density_grid_end) = pickm.load_pickle(pickle_name_undrogued)
except FileNotFoundError:
    X, Y, density_grid_end = tb.get_density_grid(ds.end_lat.values, ds.end_lon.values, xlim=xlim,
                                             ylim=ylim, latlon_box_size=latlon_box_size)
    pickm.dump_pickle((X, Y, density_grid_end), pickle_name_undrogued)


# %% Plot
def plot_global_density(X, Y, density_grid, xlim=None, ylim=None, scatter=False, latitude=None,
                        longitude=None, title=None, ax=None, legend=False, text='',
                        crs=ccrs.PlateCarree()):
    if ax is None:
        fig = plt.figure(figsize=(14, 9), dpi=300)
        ax = plt.axes(projection=crs)

    ax.set_global()
    ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.coastlines(resolution='110m', color='lightgrey', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgrey')

    gl = ax.gridlines(draw_labels=True)
    gl.xlines = False
    gl.ylines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = gl.right_labels = False

    # im = ax.pcolormesh(X, Y, density_grid, shading='nearest', cmap='hot_r', norm=LogNorm(), transform=crs)
    im = ax.scatter(X.ravel(), Y.ravel(), s=2, c=density_grid.ravel(), cmap='viridis', norm=LogNorm(), transform=crs)

    if scatter:
        ax.scatter(longitude, latitude, s=0.1, color=(0.5, 0.5, 1), transform=crs, label='Deployment locations',
                   edgecolors=None)

    if legend:
        legend = ax.legend(loc='upper right', frameon=True)
        legend.legendHandles[0]._sizes = [50]  # adjust as needed

    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax,
                        ticks=[0, 10, 100, 1000, 10000], shrink=0.9)
    cbar.set_label('Density')

    if title is not None:
        ax.set_title(title)

    # Add text in top left corner
    ax.text(0.01, 0.95, text, transform=ax.transAxes, fontsize=14, verticalalignment='top',
        bbox=dict(facecolor='lightgrey', alpha=0.75, edgecolor='darkgrey'))


crs = ccrs.PlateCarree()


fig = plt.figure(figsize=(10, 9), dpi=300)
ax1 = fig.add_subplot(3, 1, 1, projection=crs)
ax2 = fig.add_subplot(3, 1, 2, projection=crs)
ax3 = fig.add_subplot(3, 1, 3, projection=crs)

# get the current position of each axes
pos1 = ax1.get_position()
pos2 = ax2.get_position()
pos3 = ax3.get_position()

# set a new position
new_pos1 = [pos1.x0, pos1.y0, pos2.width, pos1.height]
new_pos2 = [pos2.x0, pos2.y0, pos2.width, pos2.height]
new_pos3 = [pos3.x0, pos3.y0, pos2.width, pos3.height]

ax1.set_position(new_pos1)
ax2.set_position(new_pos2)
ax3.set_position(new_pos3)

plot_global_density(X, Y, density_grid_deploy, title='Deployment', crs=ccrs.PlateCarree(),
                            ylim=ylim, ax=ax1)
plot_global_density(X, Y, density_grid_lost, title='Lost drogue', crs=ccrs.PlateCarree(),
                            ax=ax2, ylim=ylim)
plot_global_density(X, Y, density_grid_end, ylim=ylim, title='Grounded', crs=ccrs.PlateCarree(),
                            ax=ax3)

plt.tight_layout()

plt.savefig(f'figures/density_deploy_lost_beach_deathtype_{type_death}_{percentage}_{latlon_box_size}.png', dpi=300)

plt.show()