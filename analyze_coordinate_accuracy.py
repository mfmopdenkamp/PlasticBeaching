# Analyze the accuracy of the coordinate system used in the Global Drifter Program data. The coordinate systems are GPS
# and Argos. The GPS coordinate system is more accurate than the Argos coordinate system. The GPS coordinate system is
# based on the WGS84 ellipsoid, while the Argos coordinate system is based on the WGS72 ellipsoid. The difference between
# the two coordinate systems is about 1.5 km in the north-south direction and 1 km in the east-west direction. The
# difference is small enough that it is not a problem for the purposes of this project.

import numpy as np
import matplotlib.pyplot as plt
import load_data
import analyzer as a

ds = load_data.get_ds_drifters('gdp_v2.00.nc_no_sst')

traj_gps = np.where(ds.location_type.values)[0]
obs_gps = a.obs_from_traj(ds, traj_gps)
ds_gps = ds.isel(traj=traj_gps, obs=obs_gps)

traj_argos = np.where(np.invert(ds.location_type.values))[0]
obs_argos = a.obs_from_traj(ds, traj_argos)
ds_argos = ds.isel(traj=traj_argos, obs=obs_argos)


def get_std_err(ds):
    lat_err = ds.err_lat.values[np.where(ds.err_lat > 0)[0]]
    lon_err = ds.err_lon.values[np.where(ds.err_lon > 0)[0]]


    return np.sqrt(np.mean(lat_err**2 + lon_err**2))


lat_err_mean_gps = ds_gps.err_lat.values[np.where(ds_gps.err_lat > 0)[0]].mean()
lat_err_std_gps = ds_gps.err_lat.values[np.where(ds_gps.err_lat > 0)[0]].std()
lon_err_mean_gps = ds_gps.err_lon.values[np.where(ds_gps.err_lon > 0)[0]].mean()
lon_err_std_gps = ds_gps.err_lon.values[np.where(ds_gps.err_lon > 0)[0]].std()

lat_err_mean_argos = ds_argos.err_lat.values[np.where(ds_argos.err_lat > 0)[0]].mean()
lat_err_std_argos = ds_argos.err_lat.values[np.where(ds_argos.err_lat > 0)[0]].std()
lon_err_mean_argos = ds_argos.err_lon.values[np.where(ds_argos.err_lon > 0)[0]].mean()
lon_err_std_argos = ds_argos.err_lon.values[np.where(ds_argos.err_lon > 0)[0]].std()
