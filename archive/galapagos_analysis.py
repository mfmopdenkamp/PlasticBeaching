import picklemanager as pickm
from create_subset_shoreline_proximity import create_subset
import toolbox as tb
from plotter import *
import matplotlib.pyplot as plt

min_lon = -92.5
max_lon = -88.5
min_lat = -1.75
max_lat = 0.75


def load_subset(min_lon, max_lon, min_lat, max_lat):
    proximity = 10
    ds = create_subset(proximity)

    lat = (ds.latitude <= max_lat) & (ds.latitude >= min_lat)
    lon = (ds.longitude <= max_lon) & (ds.longitude >= min_lon)
    obs = np.where(lat*lon)[0]
    traj = np.where(np.isin(ds.ID, np.unique(ds.ids.isel(obs=obs))))[0]
    ds_g = ds.sel(obs=obs, traj=traj)
    return ds_g


pickle_name = 'ds_galapagos'
ds_g = pickm.pickle_wrapper(pickle_name, load_subset, min_lon, max_lon, min_lat, max_lat)

df_shore = load_data.get_shoreline(resolution='f')

selected_gdf = df_shore[(df_shore.bounds['minx'] >= min_lon) & (df_shore.bounds['maxx'] <= max_lon) &
                  (df_shore.bounds['miny'] >= min_lat) & (df_shore.bounds['maxy'] <= max_lat)]


def add_distance_shoreline(ds):
    ds['distance_shoreline'] = ('obs', tb.find_shortest_distance(ds, selected_gdf))
    return ds


ds_g = pickm.pickle_wrapper('ds_galapagos_distance', add_distance_shoreline, ds_g)


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


plot_distances(ds_g)


#%%
plot_last_distances(ds_g)


def plot_many(ds):
    
    plot_velocity_hist(ds)

    plot_distance_hist(ds)
    
    plot_velocity_distance(ds)
    
    plot_death_type_bar(ds)
    
    
#%% select only death type == 1
death_type = 1
traj = np.where(ds_g.type_death == death_type)[0]
obs = np.where(np.isin(ds_g.ids, ds_g.ID[traj]))[0]
ds_g_1 = ds_g.isel(traj=np.where(ds_g.type_death == death_type)[0], obs=obs)

plot_many(ds_g_1)
plot_distances(ds_g_1)
plot_last_distances(ds_g_1)

#%% select only death type == 3
death_type = 3
traj = np.where(ds_g.type_death == death_type)[0]
obs = np.where(np.isin(ds_g.ids, ds_g.ID[traj]))[0]
ds_g_3 = ds_g.isel(traj=np.where(ds_g.type_death == death_type)[0], obs=obs)

plot_many(ds_g_3)
plot_distances(ds_g_3)
plot_last_distances(ds_g_3)