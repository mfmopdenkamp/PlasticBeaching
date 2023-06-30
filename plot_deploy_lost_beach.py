from tqdm import tqdm
import numpy as np
import load_data
import picklemanager
import picklemanager as pickm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import toolbox as tb
import plotter


type_death = 1
ds = pickm.load_pickle(pickm.create_pickle_path('gdp_drogue_presence'))
ds = load_data.load_subset(type_death=type_death, ds=ds)


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
pickle_name = pickm.create_pickle_path(f'density_deploy_{percentage}_{latlon_box_size}_type_death_{type_death}')
try:
    (X, Y, density_grid) = picklemanager.load_pickle(pickle_name)
except FileNotFoundError:

    X, Y, density_grid = tb.get_density_grid(ds.deploy_lat.values, ds.deploy_lon.values, ylim=ylim, xlim=xlim,
                                             latlon_box_size=latlon_box_size)

    pickm.dump_pickle((X, Y, density_grid), pickle_name)

plotter.plot_global_density(X, Y, density_grid, title='Deployment', crs=ccrs.PlateCarree(),
                            ylim=ylim, ax=fig.add_subplot(3, 1, 1, projection=crs))

# %% Plot coordinates where drogue was lost
pickle_name = pickm.create_pickle_path(f'density_lost_drogue_{percentage}_{latlon_box_size}_type_death_{type_death}')
try:
    (X, Y, density_grid) = pickm.load_pickle(pickle_name)
except FileNotFoundError:
    X, Y, density_grid = tb.get_density_grid(ds.latitude_drogue_lost, ds.longitude_drogue_lost, xlim=xlim,
                                             ylim=ylim, latlon_box_size=latlon_box_size)
    pickm.dump_pickle((X, Y, density_grid), pickle_name)

plotter.plot_global_density(X, Y, density_grid, title='Lost drogue', crs=ccrs.PlateCarree(),
                            ax=fig.add_subplot(3, 1, 2, projection=crs))

# %% Plot undrogued

pickle_name_undrogued = pickm.create_pickle_path(f'density_end_{percentage}_{latlon_box_size}_type_death_{type_death}')
try:
    (X, Y, density_grid) = pickm.load_pickle(pickle_name_undrogued)
except FileNotFoundError:
    X, Y, density_grid = tb.get_density_grid(ds.lat_end.values, ds.lon_end.values, xlim=xlim,
                                             ylim=ylim, latlon_box_size=latlon_box_size)
    pickm.dump_pickle((X, Y, density_grid), pickle_name_undrogued)

plotter.plot_global_density(X, Y, density_grid, ylim=ylim, title='Undrogued',
                            ax=fig.add_subplot(3, 1, 3, projection=crs))

plt.savefig(f'figures/density_deploy_lost_beach_deathtype_{type_death}_{percentage}_{latlon_box_size}.png', dpi=300)

plt.show()
