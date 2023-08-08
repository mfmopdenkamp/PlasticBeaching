import picklemanager as pickm
import toolbox as tb
from plotter import *
import matplotlib.pyplot as plt

min_lon = 124
max_lon = 128
min_lat = 9
max_lat = 14

ds = pickm.load_pickle(pickm.create_pickle_path('gdp_drogue_presence'))

df_shore = pickm.load_pickle(pickm.create_pickle_path('df_shoreline_f_lonlat.pkl'))


shortest_dists = tb.get_shortest_distances_v2(ds, df_shore)


ds = pickm.pickle_wrapper('ds_galapagos_distance', add_distance_shoreline, ds)


def plot_distances(ds):
    fig, ax = get_sophie_subplots(extent=(-92, -88.5, -1.75, 1), title=f'Trajectory drifter')
    pcm = ax.scatter(ds.longitude, ds.latitude, transform=ccrs.PlateCarree(),
                     c=ds.distance_shoreline, cmap='inferno')
    ax.scatter(ds.longitude, ds.latitude, transform=ccrs.PlateCarree(), c='k', s=0.4,
               label='all', alpha=0.8)
    df_shore.plot(ax=ax)
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label('Distance to shore')
    ax.set_ylabel('Latitude N')
    ax.set_xlabel('Longitude E')
    plt.show()


plot_distances(ds)


#%%
plot_last_distances(ds)


def plot_many(ds):
    
    plot_velocity_hist(ds)

    plot_distance_hist(ds)
    
    plot_velocity_distance(ds)
    
    plot_death_type_bar(ds)
    
    
#%% select only death type == 1
death_type = 1
traj = np.where(ds.type_death == death_type)[0]
obs = np.where(np.isin(ds.ids, ds.ID[traj]))[0]
ds_g_1 = ds.isel(traj=np.where(ds.type_death == death_type)[0], obs=obs)

plot_many(ds_g_1)
plot_distances(ds_g_1)
plot_last_distances(ds_g_1)

#%% select only death type == 3
death_type = 3
traj = np.where(ds.type_death == death_type)[0]
obs = np.where(np.isin(ds.ids, ds.ID[traj]))[0]
ds_g_3 = ds.isel(traj=np.where(ds.type_death == death_type)[0], obs=obs)

plot_many(ds_g_3)
plot_distances(ds_g_3)
plot_last_distances(ds_g_3)