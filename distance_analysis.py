import load_data
import picklemanager as pickm
import numpy as np
import matplotlib.pyplot as plt

pickle_name_undrogued = pickm.create_pickle_ds_gdp_name(gps_only=True, undrogued_only=True)
ds_prev = pickm.load_pickle(pickm.create_pickle_path(pickle_name_undrogued))

n_obs = len(ds_prev.obs)
n_traj = len(ds_prev.traj)

distances = np.flip(np.arange(5, 1000, 100))
trajs = np.zeros(len(distances))
obss = np.zeros(len(distances))

fractions_traj = np.zeros(len(distances))
fractions_obs = np.zeros(len(distances))

velocities = np.zeros(len(distances))


for i, distance in enumerate(distances):
    ds = load_data.load_subset(max_aprox_distance_km=distance, ds=ds_prev)

    trajs[i] = len(ds.traj)
    obss[i] = len(ds.obs)

    fractions_traj[i] = len(ds.traj) / n_traj
    fractions_obs[i] = len(ds.obs) / n_obs
    velocities[i] = np.hypot(ds.ve.values, ds.vn.values).mean()

    ds_prev = ds

#%%
fig1, ax = plt.subplots()
ax.plot(distances, fractions_traj*100, c='b', label='Trajectories')
ax.plot(distances, fractions_obs*100, c='r', label='Observations')

ax.set_ylabel('Fraction trajectories (%)', color='b')
ax.set_ylabel('Fraction observations (%)', color='r')
ax.set_title('Fractions within distance to the shoreline')
ax.legend(loc='upper left')
ax.grid(True)

plt.savefig('figures/distance_percentage_traj_obs_0.png')

plt.show()


#%%
# plot results
fig2, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
ax[0].plot(distances, fractions_traj*100, c='b', label='Trajectories')
ax[0].plot(distances, fractions_obs*100, c='r', label='Observations')

ax[0].set_ylabel('Fraction trajectories (%)', color='b')
ax[0].set_ylabel('Fraction observations (%)', color='r')
ax[0].set_title('Fractions within distance to the shoreline')
ax[0].legend(loc='upper left')
ax[0].grid(True)


ax[1].plot(distances, velocities, c='g', label='Mean velocity')
ax[1].set_ylabel('Mean velocity (m/s)', color='g')
ax[1].set_xlabel('Distance threshold (km)')
ax[1].grid(True)

plt.savefig('figures/distance_percentage_traj_obs_0.png')

plt.show()
