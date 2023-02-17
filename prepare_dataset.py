import load_data
import pickle_manager as pickm
from create_subset import create_subset
import analyzer
import numpy as np
from plotter import *
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    ds['distance_shoreline'] = ('obs', analyzer.find_shortest_distance(ds, selected_gdf))
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
    
    fig, ax = plt.subplots()
    ax.hist(np.hypot(ds.ve, ds.vn), bins=50)
    ax.set_xlabel('velocity [m/s]')
    ax.set_ylabel('# data points')
    plt.show()

    fig, ax = plt.subplots()
    ax.hist(ds.distance_shoreline, bins=50)
    ax.set_xlabel('distance to the shoreline [m]')
    ax.set_ylabel('# data points')
    plt.show()
    
    fig, ax = plt.subplots()
    ax.scatter(ds.distance_shoreline, np.hypot(ds.ve, ds.vn))
    ax.set_ylabel('velocity [m/s]')
    ax.set_xlabel('distance to the shoreline [m]')
    plt.semilogy()
    plt.show()
    
    death_types = np.unique(ds.type_death)
    n_death_types = np.zeros(len(death_types))
    for i_death, death_type in enumerate(death_types):
        n_death_types[i_death] = sum(ds.type_death == death_type)
    
    fig, ax = plt.subplots()
    ax.bar(death_types, n_death_types)
    ax.set_xlabel('death type')
    ax.set_ylabel('# drifters')
    plt.xticks(death_types)
    plt.show()
    
    
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