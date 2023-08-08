import load_data
import picklemanager as pickm
import toolbox as tb
import plotter

# In[11]:
percentage = 100
ylim = [-88, 88]
xlim = None
latlon_box_size = 1


# %% Plot drogued
pickle_name = pickm.create_pickle_path(f'density_plot_drogued_{percentage}_{latlon_box_size}')
try:
    (X_drog, Y_drog, density_grid_drog) = pickm.load_pickle(pickle_name)
except FileNotFoundError:
    name_drogued = pickm.create_pickle_ds_gdp_name(percentage=percentage, drogued=True)
    ds_drogued = pickm.pickle_wrapper(name_drogued, load_data.load_subset, drogued=True,
                                  traj_percentage=percentage)

    X_drog, Y_drog, density_grid_drog = tb.get_density_grid(ds_drogued.latitude.values, ds_drogued.longitude.values, xlim=xlim,
                                             ylim=ylim, latlon_box_size=latlon_box_size)
    pickm.dump_pickle((X_drog, Y_drog, density_grid_drog), pickle_name)



# %% Plot undrogued

pickle_name_undrogued = pickm.create_pickle_path(f'density_plot_undrogued_{percentage}_{latlon_box_size}')
try:
    (X_undrog, Y_undrog, density_grid) = pickm.load_pickle(pickle_name_undrogued)
except FileNotFoundError:
    name_undrogued = pickm.create_pickle_ds_gdp_name(percentage=percentage, drogued=False)
    ds_undrogued = pickm.pickle_wrapper(name_undrogued, load_data.load_subset, drogued=False,
                                        traj_percentage=percentage)

    X_undrog, Y_undrog, density_grid = tb.get_density_grid(ds_undrogued.latitude.values, ds_undrogued.longitude.values, xlim=xlim,
                                             ylim=ylim, latlon_box_size=latlon_box_size)
    pickm.dump_pickle((X_undrog, Y_undrog, density_grid), pickle_name_undrogued)

#%%
ds = pickm.load_pickle(pickm.create_pickle_path('ds_gdp_drogued'))
lons = ds.deploy_lon.values
lats = ds.deploy_lat.values
#%% Plot both
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

crs = ccrs.PlateCarree()


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
    ax.coastlines(resolution='50m', color='black', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgrey')

    gl = ax.gridlines(draw_labels=True)
    gl.xlines = False
    gl.ylines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = gl.right_labels = False

    im = ax.pcolormesh(X, Y, density_grid, shading='nearest', cmap='hot_r', norm=LogNorm(), transform=crs)
    if scatter:
        ax.scatter(longitude, latitude, s=0.5, color=(0.5, 0.5, 1), transform=crs, label='Deployment locations', alpha=0.5)

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


fig = plt.figure(figsize=(10, 9), dpi=300)
ax1 = fig.add_subplot(2, 1, 1, projection=crs)
ax2 = fig.add_subplot(2, 1, 2, projection=crs)

# get the current position of each axes
pos1 = ax1.get_position()
pos2 = ax2.get_position()

# set a new position
new_pos1 = [pos1.x0-2, pos1.y0, pos2.width, pos1.height]
new_pos2 = [pos2.x0, pos2.y0, pos2.width, pos2.height]

ax1.set_position(new_pos1)
ax2.set_position(new_pos2)

# Then you plot your data using these axes
plot_global_density(X_drog, Y_drog, density_grid_drog, title='Drogued', crs=crs,
                            ax=ax1, scatter=True, longitude=lons, latitude=lats, legend=True, text='a)')
plot_global_density(X_undrog, Y_undrog, density_grid, ylim=ylim, title='Undrogued', crs=crs,
                            ax=ax2, text='b)')

plt.tight_layout()

plt.savefig(f'figures/density_drogued_undrogued_{percentage}_{latlon_box_size}.png', dpi=300)

plt.show()



