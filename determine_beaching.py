import pickle_manager
import pickle_manager as pickm
import numpy as np
from plotter import *

pickle_name = pickle_manager.create_pickle_name('ds_galapagos_distance')
ds_g = pickm.load_pickle(pickle_name)


trapping_rows = np.empty(0, dtype=int)
for ID in ds_g.ID:
    rows = np.where(ds_g.ids == ID)[0]
    distance = ds_g.distance_shoreline[rows]
    velocity = np.hypot(ds_g.vn[rows], ds_g.ve[rows])

    count = 0
    threshold_h = 4
    for i, (d, v) in enumerate(zip(distance, velocity)):
        if d < 500 and v < 0.1:
            count += 1
        else:
            if count >= threshold_h:
                trapping_rows = np.append(trapping_rows, rows[i-count:i])
            count = 0


def plot_distances(ds):
    fig, ax = get_sophie_subplots(extent=(-92, -88.5, -1.75, 1), title=f'Trajectory drifter')
    ds = ds.isel(obs=trapping_rows)
    pcm = ax.scatter(ds.longitude, ds.latitude, transform=ccrs.PlateCarree(),
                     c=ds.distance_shoreline, cmap='inferno')
    ax.scatter(ds.longitude, ds.latitude, transform=ccrs.PlateCarree(), c='k', s=0.4,
               label='all', alpha=0.8)
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label('Distance to shore')
    ax.set_ylabel('Latitude N')
    ax.set_xlabel('Longitude E')
    plt.show(bbox_inches='tight')


plot_distances(ds_g)


traj = np.where(np.isin(ds_g.ID, np.unique(ds_g.ids.isel(obs=trapping_rows))))[0]
ds_g_trap = ds_g.isel(obs=trapping_rows, traj=traj)

death_types = np.unique(ds_g_trap.type_death)
n_death_types = np.zeros(len(death_types))
for i_death, death_type in enumerate(death_types):
    n_death_types[i_death] = sum(ds_g_trap.type_death == death_type)

fig, ax = plt.subplots()
ax.bar(death_types, n_death_types)
ax.set_xlabel('death type')
ax.set_ylabel('# drifters')
plt.xticks(death_types)
plt.show()