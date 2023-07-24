# Analyze the accuracy of the coordinate system used in the Global Drifter Program data. The coordinate systems are GPS
# and Argos. The GPS coordinate system is more accurate than the Argos coordinate system. The GPS coordinate system is
# based on the WGS84 ellipsoid, while the Argos coordinate system is based on the WGS72 ellipsoid. The difference between
# the two coordinate systems is about 1.5 km in the north-south direction and 1 km in the east-west direction. The
# difference is small enough that it is not a problem for the purposes of this project.

import numpy as np
import matplotlib.pyplot as plt
import load_data
import math
from tqdm import tqdm
from scipy import stats


ds = load_data.get_ds_drifters('gdp_v2.00.nc_no_sst')
ds_gps = load_data.load_subset(location_type='gps', ds=ds)
ds_argos = load_data.load_subset(location_type='argos', ds=ds)

#%%
def get_std_err(ds):
    lat_err = ds.err_lat.values[ds.err_lat > 0]
    lon_err = ds.err_lon.values[ds.err_lon > 0]

    return np.sqrt(np.mean(lat_err**2 + lon_err**2))


lat_err_mean_gps = ds_gps.err_lat.values[ds_gps.err_lat > 0].mean()
lat_err_std_gps = ds_gps.err_lat.values[ds_gps.err_lat > 0].std()
lon_err_mean_gps = ds_gps.err_lon.values[ds_gps.err_lon > 0].mean()
lon_err_std_gps = ds_gps.err_lon.values[ds_gps.err_lon > 0].std()

lat_err_mean_argos = ds_argos.err_lat.values[ds_argos.err_lat > 0].mean()
lat_err_std_argos = ds_argos.err_lat.values[ds_argos.err_lat > 0].std()
lon_err_mean_argos = ds_argos.err_lon.values[ds_argos.err_lon > 0].mean()
lon_err_std_argos = ds_argos.err_lon.values[ds_argos.err_lon > 0].std()


#%%
def get_error_distance(ds):

    mask = (ds.err_lat.values > 0) & (ds.err_lon.values > 0) & (ds.err_lat.values < 45) & (ds.err_lon.values < 45)

    lats = ds.latitude.values[mask] * math.pi / 180
    err_lons = ds.err_lon.values[mask]
    err_lats = ds.err_lat.values[mask]

    dy = err_lats * 111132.92
    dx = err_lons * 111132.92 * np.cos(lats)
    err_dist = np.hypot(dx, dy)

    return err_dist


err_dist_gps = get_error_distance(ds_gps)
err_dist_argos = get_error_distance(ds_argos)

print(f'GPS error distance mean and std: {err_dist_gps.mean()}, {err_dist_gps.std()}')
print(f'Argos error distance mean and std: {err_dist_argos.mean()}, {err_dist_argos.std()}')
#%%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=300, sharey='row')
bins = 100

# Plot for GPS
axs[0, 0].hist(err_dist_gps, bins=bins, range=(0, 8000))
axs[0, 0].set_yscale('log')
axs[0, 0].set_xlabel('Error distance (m)')
axs[0, 0].set_ylabel('Number of observations')
axs[0, 0].set_title('GPS')
axs[0, 0].text(0.9, 0.9, 'a)', transform=axs[0, 0].transAxes, fontsize=12, va='top')

# Plot for Argos
axs[0, 1].hist(err_dist_argos, bins=bins, range=(0, 8000))
axs[0, 1].set_yscale('log')
axs[0, 1].set_xlabel('Error distance (m)')
axs[0, 1].set_title('Argos')
axs[0, 1].text(0.9, 0.9, 'b)', transform=axs[0, 1].transAxes, fontsize=12, va='top')

# Plot with shorter range for GPS
axs[1, 0].hist(err_dist_gps, bins=bins, range=(0, 200))
axs[1, 0].set_yscale('log')
axs[1, 0].set_xlabel('Error distance (m)')
axs[1, 0].set_ylabel('Number of observations')
axs[1, 0].text(0.9, 0.9, 'c)', transform=axs[1, 0].transAxes, fontsize=12, va='top')

# Plot with shorter range for Argos
axs[1, 1].hist(err_dist_argos, bins=bins, range=(0, 200))
axs[1, 1].set_yscale('log')
axs[1, 1].set_xlabel('Error distance (m)')
axs[1, 1].text(0.9, 0.9, 'd)', transform=axs[1, 1].transAxes, fontsize=12, va='top')

plt.tight_layout()

plt.savefig('figures/coordinates_accuracy.png', dpi=300)

plt.show()


