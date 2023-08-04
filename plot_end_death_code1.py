from numba import jit
import numpy as np
import picklemanager as pickm

R_earth = 6371  # km
ds = pickm.load_pickle(pickm.create_pickle_path('ds_gdp_death1'))
ds['speed'] = np.hypot(ds.ve.values, ds.vn.values)

#%%
@jit(nopython=True)
def process_trajectories(traj_values, type_death_values, traj_idx, undrogue_presence, velocity_values,
                         latitude_values, longitude_values, aprox_dists):
    n = np.sum(type_death_values == 1)
    final_segments_drogueness = np.empty((n, 25))
    final_segments_velocity = np.empty((n, 25))
    final_segments_latitude = np.empty((n, 25))
    final_segments_longitude = np.empty((n, 25))
    final_segments_aprox_dist = np.empty((n, 25))

    count = 0
    rows_to_drop = []
    for j, death_type in zip(traj_values, type_death_values):
        if death_type == 1:
            start = traj_idx[j + 1] - 25
            if start < traj_idx[j]:
                # if the trajectory is shorter than 24 hours, fill whole row with NaNs
                rows_to_drop.append(count)
            else:
                slc = slice(start, traj_idx[j + 1])
                final_segments_drogueness[count] = undrogue_presence[slc]
                final_segments_velocity[count] = velocity_values[slc]
                final_segments_latitude[count] = latitude_values[slc]
                final_segments_longitude[count] = longitude_values[slc]
                final_segments_aprox_dist[count] = aprox_dists[slc]
            count += 1

    return final_segments_drogueness, final_segments_velocity, final_segments_latitude,\
           final_segments_longitude, final_segments_aprox_dist, rows_to_drop


traj_idx = np.insert(np.cumsum(ds.rowsize.values), 0, 0)

final_segments_drogueness, final_segments_velocity, final_segments_latitude,\
final_segments_longitude, final_segments_aprox_dist, rows_to_drop = process_trajectories(
    ds.traj.values, ds.type_death.values, traj_idx, ds.drogue_presence.values, ds.speed.values,
    ds.latitude.values, ds.longitude.values, ds.aprox_distance_shoreline.values)
final_segments_drogueness = np.delete(final_segments_drogueness, rows_to_drop, axis=0)
final_segments_velocity = np.delete(final_segments_velocity, rows_to_drop, axis=0)
final_segments_latitude = np.delete(final_segments_latitude, rows_to_drop, axis=0)
final_segments_longitude = np.delete(final_segments_longitude, rows_to_drop, axis=0)
final_segments_aprox_dist = np.delete(final_segments_aprox_dist, rows_to_drop, axis=0)
#%% calc real final distance
final_segments_real_distance = np.zeros(final_segments_aprox_dist.shape[0])
df_shore = pickm.load_pickle(pickm.create_pickle_path('df_shoreline_f_lonlat'))


@jit(nopython=True)
def calc_real_distance(final_segments_real_distance, lons_state, lats_state,
                       lons_shore, lats_shore):

    lons_state = np.radians(lons_state)
    lats_state = np.radians(lats_state)
    lons_shore = np.radians(lons_shore)
    lats_shore = np.radians(lats_shore)

    # Strangely, somehow sometimes the shore is not found in the box around the coordinate!
    no_near_shore_mask = np.zeros(len(lats_state), dtype=np.bool_)
    R_earth = 6371.0  # km
    # Loop over all segments
    for i_state, (lon, lat) in enumerate(zip(lons_state, lats_state)):

        # Get shore points in a box around the coordinate
        side_length = 4 * np.sqrt(2) * 24  # km

        lon_length = side_length / (R_earth * np.cos(lat))  # adjust for longitude
        min_lon = lon - lon_length
        if min_lon < -np.pi:
            min_lon += 2 * np.pi
        max_lon = lon + lon_length
        if max_lon > np.pi:
            max_lon -= 2 * np.pi

        lat_length = side_length / R_earth
        min_lat = lat - lat_length
        max_lat = lat + lat_length

        mask_lon = (lons_shore >= min_lon) & (lons_shore <= max_lon)
        mask_lat = (lats_shore >= min_lat) & (lats_shore <= max_lat)
        mask = mask_lon & mask_lat

        if np.sum(mask):
            lon_shore_state = lons_shore[mask]
            lat_shore_state = lats_shore[mask]

            dlon = lon_shore_state - lon
            dlat = lat_shore_state - lat

            dxs = R_earth * np.cos(lat) * dlon
            dys = R_earth * dlat

            distances_shore_state = np.sqrt(dxs ** 2 + dys ** 2)

            # Save the shortest distance and direction to the shore
            # - distance:
            shortest_distance = np.min(distances_shore_state)
            final_segments_real_distance[i_state] = shortest_distance  # 'shortest_distance'
        else:
            no_near_shore_mask[i_state] = True

    return final_segments_real_distance, no_near_shore_mask


final_segments_real_distance, no_near_shore_mask = calc_real_distance(final_segments_real_distance,
                                                                      final_segments_longitude[:, -1],
                                                                      final_segments_latitude[:, -1],
                                                                      df_shore.longitude.values,
                                                                      df_shore.latitude.values)

#%%
diff_drogueness = final_segments_drogueness[:, :-1] - final_segments_drogueness[:, 1:]
mask_drogue_lost = np.sum(diff_drogueness, axis=1).astype(np.bool_)
count_lost_drogues = np.sum(mask_drogue_lost)

sum_drogueness = np.sum(final_segments_drogueness,axis=1)
mask_drogued = sum_drogueness == 25
mask_undrogued = sum_drogueness == 0
count_undrogued = np.sum(mask_undrogued)
count_drogued = np.sum(mask_drogued)


#%% Plot three last distance distribution for drogued undrogued and lost drogue segments
fd_drogued = final_segments_real_distance[mask_drogued & ~no_near_shore_mask]
fd_undrogued = final_segments_real_distance[mask_undrogued & ~no_near_shore_mask]
fd_lost_drogue = final_segments_real_distance[mask_drogue_lost & ~no_near_shore_mask]

import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 1, figsize=(8, 10))

data = [fd_drogued, fd_undrogued, fd_lost_drogue]
labels = ['Drogued', 'Undrogued', 'Lost Drogue']
bins = [np.arange(0,220,20), np.arange(0, 52, 2), np.arange(0, 52, 2)]

for i, ax in enumerate(axs):
    ax.hist(data[i], bins=bins[i], alpha=0.5, edgecolor='black')
    ax.set_title(labels[i])
    ax.set_ylabel('Number of drifters')
    ax.grid(True)

axs[-1].set_xlabel('Distance from shoreline (km)')


plt.tight_layout()
plt.savefig('figures/last_3_dist_drogued_undrogued_lost_drogue.png', dpi=300, bbox_inches='tight')

plt.show()




#%% plot final segments
import matplotlib.pyplot as plt

# Define the time range for the last 24 hours
time_range = np.arange(-24, 1, 1)

# Calculate mean and standard deviation for each variable at each time point
mean_drogue = np.mean(final_segments_drogueness, axis=0)
std_drogue = np.std(final_segments_drogueness, axis=0)

mean_velocity = np.mean(final_segments_velocity, axis=0)
std_velocity = np.std(final_segments_velocity, axis=0)

final_segments_aprox_dist_diff = final_segments_aprox_dist[:, :-1] - final_segments_aprox_dist[:, 1:]
final_segments_aprox_dist_to_last = final_segments_aprox_dist - final_segments_aprox_dist[:, -1][:, None]
#drop the row with high zscore in diff
from scipy.stats import zscore
z_scores = np.abs(zscore(final_segments_aprox_dist_diff))
rows_to_remove = np.any(z_scores > 6, axis=1)
final_segments_aprox_dist_to_last = final_segments_aprox_dist_to_last[~rows_to_remove]

mean_aprox_dist_to_last = np.mean(final_segments_aprox_dist_to_last, axis=0)
std_aprox_dist_to_last = np.std(final_segments_aprox_dist_to_last, axis=0)

# Create a figure with 3 subplots, one for each variable
fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

# Plot mean and standard deviation for each variable
axs[0].plot(time_range, mean_drogue*100, label='Mean')
# axs[0].fill_between(time_range, mean_drogue-std_drogue, mean_drogue+std_drogue, color='b', alpha=0.2, label='Std dev')
axs[1].plot(time_range, mean_velocity, label='Mean')
axs[1].fill_between(time_range, mean_velocity-std_velocity, mean_velocity+std_velocity, color='b', alpha=0.2,
                    label='standard deviation')
# plot aprox dist
axs[2].plot(time_range, mean_aprox_dist_to_last, label='Mean')
axs[2].fill_between(time_range, mean_aprox_dist_to_last - std_aprox_dist_to_last,
                    mean_aprox_dist_to_last + std_aprox_dist_to_last, color='b', alpha=0.2,
                    label='standard deviation')
# bp = axs[2].boxplot(final_segments_aprox_dist_to_last, positions=time_range, widths=0.8, patch_artist=True, showfliers=False)

# Add labels and titles
axs[0].set(ylabel='Drogued drifters (%)')
axs[1].set(ylabel='Speed (m/s)')
axs[2].set(xlabel='Time (hours before death)', ylabel='Difference approximate\nnearest shoreline distance (km)')

axs[0].set_xlim([-24, 0])
axs[1].set_xlim([-24, 0])
axs[2].set_xlim([-24, 0])

# Add zero line
axs[0].axhline(0, color='darkgrey', linestyle=':')
axs[1].axhline(0, color='darkgrey', linestyle=':')
axs[2].axhline(0, color='darkgrey', linestyle=':')

# Add legends
axs[0].legend()
axs[1].legend()
axs[2].legend()

# mark the panels with letters
axs[0].text(0.03, 0.8, 'a)', transform=axs[0].transAxes, size=12, weight='bold')
axs[1].text(0.03, 0.8, 'b)', transform=axs[1].transAxes, size=12, weight='bold')
axs[2].text(0.03, 0.8, 'c)', transform=axs[2].transAxes, size=12, weight='bold')

plt.savefig('figures/final_segments_death_1.png', dpi=300)

plt.show()

#%%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import plotter

n_rows = 4
fig2, axs = plt.subplots(n_rows, 4, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(14, n_rows*3), dpi=300)

# Selecting random 10 drifters
idxs = np.random.choice(range(len(final_segments_drogueness)), 20, replace=False)

for i, ax in enumerate(axs.flatten()):
    idx = idxs[i]

    # Plotting the trajectory
    lon = final_segments_longitude[idx]
    lat = final_segments_latitude[idx]
    lon_min, lon_max = np.min(lon), np.max(lon)
    lat_min, lat_max = np.min(lat), np.max(lat)

    margin = max(abs(lon_max - lon_min), abs(lat_max - lat_min)) + 0.05
    lon_mid = lon[-1]
    lat_mid = lat[-1]
    ax = plotter.get_marc_subplots(extent=[lon_mid - margin, lon_mid + margin, lat_mid - margin, lat_mid + margin],
                                      show_coastlines=False, ax=ax)

    ax.scatter(lon[1:-1], lat[1:-1], transform=ccrs.PlateCarree())
    ax.scatter(lon[0], lat[0], transform=ccrs.PlateCarree(), marker='^', color='g', s=40, label='Start')
    ax.scatter(lon[-1], lat[-1], transform=ccrs.PlateCarree(), marker='s', color='r', s=40, label='End')
    if i == 0:
        ax.legend()

plt.tight_layout()

plt.savefig('figures/trajectories_death_code_1.png', dpi=300)

plt.show()
