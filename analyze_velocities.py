# This script analyzes the velocities within the GDP hourly dataset

import load_data
import numpy as np
import analyzer as a
import matplotlib.pyplot as plt

ds = load_data.get_ds_drifters('gdp_v2.00.nc_no_sst')

traj_gps = np.where(ds.location_type.values)[0]
obs_gps = a.obs_from_traj(ds, traj_gps)
ds_gps = ds.isel(traj=traj_gps, obs=obs_gps)

traj_argos = np.where(np.invert(ds.location_type.values))[0]
obs_argos = a.obs_from_traj(ds, traj_argos)
ds_argos = ds.isel(traj=traj_argos, obs=obs_argos)

v_gps = a.get_absolute_velocity(ds_gps)
v_argos = a.get_absolute_velocity(ds_argos)

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