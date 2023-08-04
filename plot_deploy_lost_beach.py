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
latlon_box_size = 1

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
def plot_global_density(X, Y, density_grid, xlim=None, ylim=None, scatter=False, ax=None,
                        crs=ccrs.PlateCarree(), colorbar_label='', marker='o', cmap='viridis_r',
                        label=None):
    if ax is None:
        fig = plt.figure(figsize=(6, 9), dpi=300)
        ax = plt.axes(projection=crs)

    ax.set_global()
    ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    # ax.coastlines(resolution='110m', color='lightgrey', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgrey')

    gl = ax.gridlines(draw_labels=True)
    gl.xlines = False
    gl.ylines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = gl.right_labels = False


    if scatter:
        im = ax.scatter(X.ravel(), Y.ravel(), s=3, marker=marker,
                        c=density_grid.ravel(), cmap=cmap, norm=LogNorm(), transform=crs, label=label)
    else:
        im = ax.pcolormesh(X, Y, density_grid, shading='nearest', cmap='hot_r', norm=LogNorm(), transform=crs,
                           label=label)


    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.5)

    cbar.set_label(colorbar_label, labelpad=-50)

    return ax


crs = ccrs.PlateCarree()

fig, ax = plt.subplots(figsize=(14, 8), dpi=300, subplot_kw={'projection': crs})


plot_global_density(X, Y, density_grid_lost, ylim=ylim, crs=ccrs.PlateCarree(), marker='*',
                    label='Drogues lost', cmap='cool',
                    ax=ax, scatter=True, colorbar_label=f'Lost drogues (# per {latlon_box_size}x{latlon_box_size} degrees)')
plot_global_density(X, Y, density_grid_end, ylim=ylim, crs=ccrs.PlateCarree(), label='Groundings',
                    ax=ax, scatter=True, colorbar_label=f'Trajectory ends (# per {latlon_box_size}x{latlon_box_size} degrees)')

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='*', color='w', label='Drogues lost',
                          markerfacecolor=(0.5, 0.2, 0.8), markersize=13),
                   Line2D([0], [0], marker='o', color='w', label='Trajectories end',
                          markerfacecolor=(31/256,158/256,138/256), markersize=10)]

ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()

plt.savefig(f'figures/drogue_trajend_{type_death}_{percentage}_{latlon_box_size}.png', dpi=300)

plt.show()

#%% Compare lat lon between drogues lost and trajectory end

fig, ax = plt.subplots(figsize=(14, 8), dpi=300)


ax.bar(density_grid_lost.sum(axis=1))
ax.bar(density_grid_end.sum(axis=1))

plt.show()
