# This script analyzes the velocities within the GDP hourly dataset

import load_data
import numpy as np
import toolbox as tb
import picklemanager as pickm
import matplotlib.pyplot as plt

ds = load_data.get_ds_drifters('gdp_v2.00.nc')

ds_gps = load_data.load_subset(location_type='gps', ds=ds)
ds_argos = load_data.load_subset(location_type='argos', ds=ds)

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
pickle_name_undrogued = pickm.create_pickle_ds_gdp_name(location_type=True, drogued=True)
ds_undrogued = pickm.load_pickle(pickm.create_pickle_path(pickle_name_undrogued))

n_obs = len(ds_undrogued.obs)
n_traj = len(ds_undrogued.traj)

delta_km = 0.2
max_km = 10

distances = np.flip(np.arange(delta_km, max_km, delta_km))
trajs = np.zeros(len(distances))
obss = np.zeros(len(distances))

velocities = np.zeros(len(distances))

for i, distance in enumerate(distances):
    ds = load_data.load_subset(max_aprox_distance_km=distance, ds=ds_undrogued,
                               min_aprox_distance_km=distance - delta_km)

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
