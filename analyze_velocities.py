# This script analyzes the velocities within the GDP hourly dataset

import load_data
import numpy as np
import toolbox as tb
import picklemanager as pickm
import matplotlib.pyplot as plt

ds = load_data.get_ds_drifters('gdp_v2.00.nc_no_sst')

traj_gps = np.where(ds.location_type.values)[0]
obs_gps = tb.obs_from_traj(ds, traj_gps)
ds_gps = ds.isel(traj=traj_gps, obs=obs_gps)

traj_argos = np.where(np.invert(ds.location_type.values))[0]
obs_argos = tb.obs_from_traj(ds, traj_argos)
ds_argos = ds.isel(traj=traj_argos, obs=obs_argos)

v_gps = tb.get_absolute_velocity(ds_gps)
v_argos = tb.get_absolute_velocity(ds_argos)

mean_err_vn_gps = ds_gps.err_vn.values[np.where(ds_gps.err_vn > 0)[0]].mean()
std_err_vn_gps = ds_gps.err_vn.values[np.where(ds_gps.err_vn > 0)[0]].std()
mean_err_vn_argos = ds_argos.err_vn.values[np.where(ds_argos.err_vn > 0)[0]].mean()
std_err_vn_argos = ds_argos.err_vn.values[np.where(ds_argos.err_vn > 0)[0]].std()
print(f'Mean error in northward velocity of GPS = {mean_err_vn_gps:.4f}±{std_err_vn_gps:.4f} m/s.')
print(f'Mean error in northward velocity of Argos = {mean_err_vn_argos:.4f}±{std_err_vn_argos:.4f} m/s.')


# compute and print the maximum velocity
max_v_gps = v_gps.max()
max_v_argos = v_argos.max()
print(f'Maximum velocity of GPS = {max_v_gps:.4f} m/s.')
print(f'Maximum velocity of Argos = {max_v_argos:.4f} m/s.')

#%%
pickle_name_undrogued = pickm.create_pickle_ds_gdp_name(gps_only=True, undrogued_only=True)
ds_undrogued = pickm.load_pickle(pickm.create_pickle_path(pickle_name_undrogued))

n_obs = len(ds_undrogued.obs)
n_traj = len(ds_undrogued.traj)

delta_km = 1
max_km = 31

distances = np.flip(np.arange(delta_km, max_km, delta_km))
trajs = np.zeros(len(distances))
obss = np.zeros(len(distances))

velocities = np.zeros(len(distances))

for i, distance in enumerate(distances):
    ds = load_data.load_subset(max_aprox_distance_km=distance, min_aprox_distance_km=distance-delta_km, ds=ds_undrogued)

    trajs[i] = len(ds.traj)
    obss[i] = len(ds.obs)
    velocities[i] = np.hypot(ds.ve.values, ds.vn.values).mean()

#%%
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(distances, velocities, width=-delta_km*0.8, align='edge', edgecolor='k', color='b')
ax.set_xlabel('Distance to the shoreline (km)')
ax.set_ylabel('Mean velocity (m/s)')

plt.savefig(f'figures/bar_plot_mean_velocity_distance_{delta_km}_{max_km}.png')
plt.show()
