from numba import jit
import numpy as np
import picklemanager as pickm


ds = pickm.load_pickle(pickm.create_pickle_path('gdp_drogue_presence_sst'))
ds['velocity'] = np.hypot(ds.ve.values, ds.vn.values)
#%%
@jit(nopython=True)
def process_trajectories(traj_values, type_death_values, traj_idx, undrogue_presence, sst_values, velocity_values,
                         latitude_values, longitude_values):
    n = np.sum(type_death_values == 1)
    final_segments_drogueness = np.empty((n, 25))
    final_segments_sst = np.empty((n, 25))
    final_segments_velocity = np.empty((n, 25))
    final_segments_latitude = np.empty((n, 25))
    final_segments_longitude = np.empty((n, 25))

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
                final_segments_sst[count] = sst_values[slc]
                final_segments_velocity[count] = velocity_values[slc]
                final_segments_latitude[count] = latitude_values[slc]
                final_segments_longitude[count] = longitude_values[slc]
            count += 1

    return final_segments_drogueness, final_segments_sst, final_segments_velocity, final_segments_latitude,\
           final_segments_longitude, rows_to_drop


traj_idx = np.insert(np.cumsum(ds.rowsize.values), 0, 0)

final_segments_drogueness, final_segments_sst, final_segments_velocity, final_segments_latitude,\
final_segments_longitude, rows_to_drop = process_trajectories(
    ds.traj.values, ds.type_death.values, traj_idx, ds.drogue_presence.values, ds.sst.values, ds.velocity.values,
    ds.latitude.values, ds.longitude.values)
final_segments_drogueness = np.delete(final_segments_drogueness, rows_to_drop, axis=0)
final_segments_sst = np.delete(final_segments_sst, rows_to_drop, axis=0)
final_segments_velocity = np.delete(final_segments_velocity, rows_to_drop, axis=0)
final_segments_latitude = np.delete(final_segments_latitude, rows_to_drop, axis=0)
final_segments_longitude = np.delete(final_segments_longitude, rows_to_drop, axis=0)

#%%
diff_drogueness = final_segments_drogueness[:, :-1] - final_segments_drogueness[:, 1:]
count_drogueness = np.sum(diff_drogueness, axis=1)
total_lost_drogues = np.sum(count_drogueness)

sum_drogueness = np.sum(final_segments_drogueness,axis=1)
count_drogueless = np.sum(sum_drogueness == 0)

#%% plot final segments
import matplotlib.pyplot as plt

# Define the time range for the last 24 hours
time_range = np.arange(-24, 1, 1)

# remove rows with NaNs for SST
final_segments_sst = final_segments_sst[~np.isnan(final_segments_sst).any(axis=1)]

# Calculate mean and standard deviation for each variable at each time point
mean_drogue = np.mean(final_segments_drogueness, axis=0)
std_drogue = np.std(final_segments_drogueness, axis=0)

mean_sst = np.mean(final_segments_sst, axis=0)
std_sst = np.std(final_segments_sst, axis=0)

mean_velocity = np.mean(final_segments_velocity, axis=0)
std_velocity = np.std(final_segments_velocity, axis=0)

# Create a figure with 3 subplots, one for each variable
fig, axs = plt.subplots(2, 1, figsize=(7, 8))

# Plot mean and standard deviation for each variable
axs[0].plot(time_range, mean_drogue, label='Mean')
# axs[0].fill_between(time_range, mean_drogue-std_drogue, mean_drogue+std_drogue, color='b', alpha=0.2, label='Std dev')
axs[1].plot(time_range, mean_velocity, label='Mean')
# axs[1].fill_between(time_range, mean_velocity-std_velocity, mean_velocity+std_velocity, color='b', alpha=0.2, label='Std dev')

# Add labels and titles
axs[0].set(xlabel='Time (hours before grounding)', ylabel='Drogue presence')
axs[1].set(xlabel='Time (hours before grounding)', ylabel='Velocity (m/s)')
axs[0].set_title('Drogue presence in the last 24 hours before grounding')
axs[1].set_title('Velocity in the last 24 hours before grounding')

axs[0].set_xlim([-24, 0])
axs[1].set_xlim([-24, 0])

# Add zero line
axs[0].axhline(0, color='k', linestyle='--')
axs[1].axhline(0, color='k', linestyle='--')

# Add legends
axs[0].legend()
axs[1].legend()

# Show the plot
plt.tight_layout()
plt.savefig('figures/final_segments_death_1.png', dpi=300)

plt.show()

#%%
import cartopy.crs as ccrs
import plotter

fig2, axs = plt.subplots(5, 2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 25), dpi=300)

# Selecting random 10 drifters
idxs = np.random.choice(range(len(final_segments_drogueness)), 10, replace=False)

for i, ax in enumerate(axs.flatten()):
    idx = idxs[i]

    # Plotting the trajectory
    lon = final_segments_longitude[idx]
    lat = final_segments_latitude[idx]

    margin = 0.2
    lon_mid = lon[-1]
    lat_mid = lat[-1]
    ax = plotter.get_marc_subplots(extent=[lon_mid - margin, lon_mid + margin, lat_mid - margin, lat_mid + margin],
                                      show_coastlines=False, ax=ax)

    ax.plot(lon, lat, transform=ccrs.PlateCarree())

    # Center the extent of the map around the drifter's trajectory
    lon_min, lon_max = np.min(lon), np.max(lon)
    lat_min, lat_max = np.min(lat), np.max(lat)

fig.subplots_adjust(hspace=0.1, wspace=0.1)

plt.tight_layout()

plt.savefig('figures/trajectories_death_code_1.png', dpi=300)

plt.show()
