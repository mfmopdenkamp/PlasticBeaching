import pandas as pd
import numpy as np
import load_data
import picklemanager
import picklemanager as pickm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import toolbox as tb
import plotter

# In[11]:
percentage = 100
ylim = [-88, 88]
xlim = None
latlon_box_size = 1
crs = ccrs.PlateCarree()

fig, axs = plt.subplots(3, 1, figsize=(14, 9), dpi=300)
for ax in axs.flat:
    # Plot your density map

    # Remove the extra borders
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# %% Plot total
pickle_name = pickm.create_pickle_path(f'density_plot_total_{percentage}_{latlon_box_size}')
try:
    (X, Y, density_grid) = picklemanager.load_pickle(pickle_name)
except FileNotFoundError:
    name_total = pickm.create_pickle_ds_gdp_name(percentage=percentage)
    ds = pickm.pickle_wrapper(name_total, load_data.load_subset, traj_percentage=percentage)

    X, Y, density_grid = tb.get_density_grid(ds.latitude.values, ds.longitude.values, ylim=ylim, xlim=xlim,
                                             latlon_box_size=latlon_box_size)

    pickm.dump_pickle((X, Y, density_grid), pickle_name)

plotter.plot_global_density(X, Y, density_grid, title='All', crs=ccrs.PlateCarree(),
                            ylim=ylim, ax=fig.add_subplot(3, 1, 1, projection=crs))

# %% Plot drogued
pickle_name = pickm.create_pickle_path(f'density_plot_drogued_{percentage}_{latlon_box_size}')
try:
    (X, Y, density_grid) = pickm.load_pickle(pickle_name)
except FileNotFoundError:
    name_drogued = pickm.create_pickle_ds_gdp_name(percentage=percentage, drogued=True)
    ds_drogued = pickm.pickle_wrapper(name_drogued, load_data.load_subset, drogued=True,
                                  traj_percentage=percentage)

    X, Y, density_grid = tb.get_density_grid(ds_drogued.latitude.values, ds_drogued.longitude.values, xlim=xlim,
                                             ylim=ylim, latlon_box_size=latlon_box_size)
    pickm.dump_pickle((X, Y, density_grid), pickle_name)

plotter.plot_global_density(X, Y, density_grid, title='Drogued', crs=ccrs.PlateCarree(),
                            ax=fig.add_subplot(3, 1, 2, projection=crs))

# %% Plot undrogued

pickle_name_undrogued = pickm.create_pickle_path(f'density_plot_undrogued_{percentage}_{latlon_box_size}')
try:
    (X, Y, density_grid) = pickm.load_pickle(pickle_name_undrogued)
except FileNotFoundError:
    name_undrogued = pickm.create_pickle_ds_gdp_name(percentage=percentage, drogued=False)
    ds_undrogued = pickm.pickle_wrapper(name_undrogued, load_data.load_subset, drogued=False,
                                        traj_percentage=percentage)

    X, Y, density_grid = tb.get_density_grid(ds_undrogued.latitude.values, ds_undrogued.longitude.values, xlim=xlim,
                                             ylim=ylim, latlon_box_size=latlon_box_size)
    pickm.dump_pickle((X, Y, density_grid), pickle_name_undrogued)

plotter.plot_global_density(X, Y, density_grid, ylim=ylim, title='Undrogued',
                            ax=fig.add_subplot(3, 1, 3, projection=crs))

plt.savefig(f'figures/density_drogued_undrogued_{percentage}_{latlon_box_size}.png', dpi=300)

plt.show()
