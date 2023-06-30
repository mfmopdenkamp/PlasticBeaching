import load_data
import picklemanager as pickm
import numpy as np
import tqdm
import matplotlib.pyplot as plt


ds = load_data.get_ds_drifters()
pickle_name_undrogued = pickm.create_pickle_ds_gdp_name(drogued=False)
ds_undrogued = pickm.pickle_wrapper(pickle_name_undrogued, load_data.load_subset, drogued=False, ds=ds)

pickle_name_drogued = pickm.create_pickle_ds_gdp_name(drogued=True)
ds_drogued = pickm.pickle_wrapper(pickle_name_drogued, load_data.load_subset, drogued=True, ds=ds)

n_obs_undrogued = len(ds_undrogued.obs.values)
n_obs_drogued = len(ds_drogued.obs.values)
percentage_obs_undrogued = n_obs_undrogued / (n_obs_drogued + n_obs_undrogued) * 100

n_traj_undrogued = len(ds_undrogued.traj.values)
n_traj_drogued = len(ds_drogued.traj.values)
percentage_traj_undrogued = n_traj_undrogued / (n_traj_drogued + n_traj_undrogued) * 100


#%%
# Precompute ids arrays
undrogued_ids = ds_undrogued.ids.values
drogued_ids = ds_drogued.ids.values


n_obs_drogued_d = np.zeros(len(ds.ID.values))
n_obs_undrogued_d = np.zeros(len(ds.ID.values))
for i, drifter_id in enumerate(tqdm.tqdm(ds.ID.values)):
    # Count occurrences of drifter_id in undrogued and drogued ids arrays
    n_obs_undrogued_d[i] = np.sum(undrogued_ids == drifter_id)
    n_obs_drogued_d[i] = np.sum(drogued_ids == drifter_id)

# Calculate percentage of undrogued observations
percentage_undrogued_drifter = n_obs_undrogued_d / (n_obs_drogued_d + n_obs_undrogued_d) * 100

pickm.dump_pickle(percentage_undrogued_drifter, pickm.create_pickle_path('percentage_undrogued_drifter'))
pickm.dump_pickle(n_obs_undrogued_d, pickm.create_pickle_path('n_obs_undrogued_d'))
pickm.dump_pickle(n_obs_drogued_d, pickm.create_pickle_path('n_obs_drogued_d'))

#%% Plot trajectory length histogram
bins=300
plt.figure(figsize=(10, 6))
plt.hist(ds.rowsize.values/24, bins=bins)
plt.xlabel('length of trajectory [days]')
plt.ylabel('number of drifters')

plt.savefig(f'figures/trajectory_length_histogram_{bins}bins.png', dpi=300)
plt.show()

print('Mean length of trajectory: ', np.mean(ds.rowsize.values/24))
print('Median length of trajectory: ', np.median(ds.rowsize.values/24))
print('Max length of trajectory: ', np.max(ds.rowsize.values/24))
print('Min length of trajectory: ', np.min(ds.rowsize.values/24))

#%% Plot histogram of percentage of undrogued observations per drifter
plt.figure(figsize=(10, 6))
plt.hist(percentage_undrogued_drifter, bins=100)
plt.xlabel('fraction of undrogued observations [%]')
plt.ylabel('number of drifters')

plt.yscale('log')

plt.savefig(f'figures/percentage_undrogued_histogram.png', dpi=300)

plt.show()

#%% Plot percentage of undrogued observations per drifter vs length of trajectory in heat map
percentage_undrogued = percentage_undrogued_drifter

# Compute the 2D histogram counts
hist, x_edges, y_edges = np.histogram2d(ds.rowsize.values/24, percentage_undrogued, bins=[100, 100],
                                        range=[[0, 1500], [0, 100]])

# Create a grid of x and y values
x, y = np.meshgrid(x_edges, y_edges)

# Set the maximum value for the color scale and capped values
vmax = 20
capped_value = "> " + str(vmax)

# Plot the 2D histogram as a heat map
plt.figure(figsize=(10, 6))
plt.pcolormesh(x, y, hist.T, cmap='hot_r', vmax=vmax)
plt.colorbar(label='Count')
plt.xlabel('trajectory length (hours)')
plt.ylabel('trajectory fraction undrogued (%)')

plt.show()


#%%

pickle_name = pickm.create_pickle_ds_gdp_name(location_type=True)
ds_gps = pickm.pickle_wrapper(pickle_name, load_data.load_subset, location_type=True, ds=ds)
pickle_name_undrogued = pickm.create_pickle_ds_gdp_name(location_type=True, drogued=True)
ds_gps_undrogued = pickm.pickle_wrapper(pickle_name_undrogued, load_data.load_subset, location_type=None,
                                        drogued=False, ds=ds_gps)
pickle_name_12km = pickm.create_pickle_ds_gdp_name(location_type=True, drogued=False,
                                                   max_aprox_distance_km=12)
ds_gps_undrogued_12km = pickm.pickle_wrapper(pickle_name_12km, load_data.load_subset, max_aprox_distance_km=12,
                                             ds=ds_gps_undrogued)


#%% count
n_traj_total = len(ds.traj)
n_obs_total = len(ds.obs)
obs_per_traj = n_obs_total / n_traj_total

n_obs_gps = len(ds_gps.obs)
n_traj_gps = len(ds_gps.traj)
n_obs_gps_undrogued = len(ds_gps_undrogued.obs)
n_traj_gps_undrogued = len(ds_gps_undrogued.traj)

percentage_undrogued_obs = n_obs_gps_undrogued / n_obs_gps * 100
percentage_undrogued_traj = n_traj_gps_undrogued / n_traj_gps * 100

n_obs_gps_undrogued_12km = len(ds_gps_undrogued_12km.obs)
n_traj_gps_undrogued_12km = len(ds_gps_undrogued_12km.traj)

percentage_undrogued_obs_12km = n_obs_gps_undrogued_12km / n_obs_gps_undrogued * 100
percentage_undrogued_traj_12km = n_traj_gps_undrogued_12km / n_traj_gps_undrogued * 100

obs_per_traj_gps = n_obs_gps / n_traj_gps
obs_per_traj_gps_undrogued = n_obs_gps_undrogued / n_traj_gps_undrogued
obs_per_traj_gps_undrogued_12km = n_obs_gps_undrogued_12km / n_traj_gps_undrogued_12km


#%%
print('selection\t\tnumber of observations\t\tnumber of trajectories\t\tobservations per trajectory')
print(f'GPS & Argos\t{n_obs_total}\t{n_traj_total}\t{obs_per_traj:.1f}')
print(f'GPS\t{n_obs_gps}\t{n_traj_gps}\t{obs_per_traj_gps:.1f}')
print(f'GPS, undrogued\t{n_obs_gps_undrogued}\t{n_traj_gps_undrogued}\t{obs_per_traj_gps_undrogued:.1f}')
print(f'GPS, undrogued, < 12km of coast\t{n_obs_gps_undrogued_12km}\t{n_traj_gps_undrogued_12km}\t'
        f'{obs_per_traj_gps_undrogued_12km:.1f}')
print(f'Argos\t{n_obs_total - n_obs_gps}\t{n_traj_total - n_traj_gps}\t'
    f'{(n_obs_total - n_obs_gps) / (n_traj_total - n_traj_gps):.1f}')

