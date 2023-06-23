import load_data
import matplotlib.pyplot as plt
import toolbox as tb
import pandas as pd
import numpy as np
from file_names import *
import plotter
import cartopy.crs as ccrs
crs = ccrs.PlateCarree()

ds = load_data.load_subset(type_death=1)
ds_gps = load_data.load_subset(type_death=1, location_type='gps')
df = pd.read_csv(f'data/{file_name_1}.csv', parse_dates=['time_start', 'time_end'])

print(np.unique(ds.type_death.values))
#%%

ylim= [-88, 88]
xlim=None
lonlat_box_size = 5


X, Y, density_grid_original = tb.get_density_grid(ds.end_lat.values, ds.end_lon.values, ylim=ylim, xlim=xlim, latlon_box_size=lonlat_box_size)
density_grid_gps = tb.get_density_grid(ds_gps.end_lat.values, ds_gps.end_lon.values, ylim=ylim, xlim=xlim, latlon_box_size=lonlat_box_size)[2]
density_grid_own = tb.get_density_grid(df.latitude_end.values, df.longitude_end.values, ylim=ylim, xlim=xlim, latlon_box_size=lonlat_box_size)[2]


# %%
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


plotter.plot_global_density(X, Y, density_grid_original, title='death type 1', ax=fig.add_subplot(3,1,1, projection=crs))
plotter.plot_global_density(X, Y, density_grid_gps, title='death type 1 gps', ax=fig.add_subplot(3,1,2, projection=crs))
plotter.plot_global_density(X, Y, density_grid_own, title='SDDA', ax=fig.add_subplot(3,1,3, projection=crs))

plt.savefig(f'figures/compare_end_coords_{lonlat_box_size}.png')

plt.show()