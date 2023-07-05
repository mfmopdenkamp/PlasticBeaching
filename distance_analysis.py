import load_data
import picklemanager as pickm
import numpy as np
import matplotlib.pyplot as plt

ds = pickm.load_pickle(pickm.create_pickle_path('gdp_v2.00.nc_no_sst'))

#%%
n_obs = len(ds.obs)
n_traj = len(ds.traj)


def get_fractions(distances):
    traj_far = np.zeros(len(distances))
    obs_far = np.zeros(len(distances))

    for i, distance in enumerate(distances):
        mask = ds.aprox_distance_shoreline.values < distance
        obs = ds.obs.values[mask]

        traj_far[i] = len(np.unique(ds.ids.isel(obs=obs)))
        obs_far[i] = len(obs)

    fractions_traj = traj_far / n_traj
    fractions_obs = obs_far / n_obs

    return fractions_traj, fractions_obs

distances_far = np.flip(np.arange(5, 1000, 100))
fractions_traj_far, fractions_obs_far = get_fractions(distances_far)
distances_close = np.flip(np.arange(1, 16, 1))
fractions_traj_close, fractions_obs_close = get_fractions(distances_close)


#%%
fig, axs = plt.subplots(2, 2, sharex='col', figsize=(10, 10), dpi=300)
axs[0, 0].plot(distances_far, fractions_traj_far)
axs[1, 0].plot(distances_far, fractions_obs_far)
axs[0, 1].plot(distances_close, fractions_traj_close)
axs[1, 1].plot(distances_close, fractions_obs_close)

axs[0, 0].set_ylabel('Trajectories')
axs[1, 0].set_ylabel('Observations')

fig.text(0.5, 0.04, 'Maximum distance to the shoreline (km)', ha='center')
fig.text(0.04, 0.5, 'Fraction of total', va='center', rotation='vertical')

axs[0, 0].set_title('Far from the shore')
axs[0, 1].set_title('Close to the shore')

for ax in axs.flat:
    ax.grid(True)

plt.savefig('figures/distance_percentage_traj_obs_00.png')

plt.show()