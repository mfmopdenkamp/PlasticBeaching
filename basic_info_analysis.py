import load_data
import picklemanager as pickm
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

ds = pickm.load_pickle(pickm.create_pickle_path('gdp_drogue_presence'))
print(ds.info())
#%%
n_obs_drogued = ds.drogue_presence.values.sum()
n_obs_undrogued = len(ds.obs) - n_obs_drogued
percentage_obs_undrogued = n_obs_undrogued / len(ds.obs) * 100

traj_undrogued_hours = np.zeros(len(ds.ID.values))
traj_idx = np.insert(np.cumsum(ds.rowsize.values), 0, 0)
undrogue_presence = np.invert(ds.drogue_presence.values)
for j in tqdm.tqdm(ds.traj.values):
    traj_undrogued_hours[j] = undrogue_presence[slice(traj_idx[j], traj_idx[j + 1])].sum()

drifter_lost_drogue = traj_undrogued_hours > 0
percentage_traj_lost_drogue = drifter_lost_drogue.sum() / len(ds.traj) * 100
# Calculate percentage of undrogued observations
percentage_undrogued_obs_per_drifter = traj_undrogued_hours / ds.rowsize.values * 100


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
plt.hist(percentage_undrogued_obs_per_drifter, bins=100)
plt.xlabel('fraction undrogued [%]')
plt.ylabel('number of drifters')

plt.yscale('log')

plt.savefig(f'figures/fraction_undrogued_histogram.png', dpi=300)

plt.show()

#%% Plot percentage of undrogued observations per drifter vs length of trajectory in heat map
# Compute the 2D histogram counts
hist, x_edges, y_edges = np.histogram2d(ds.rowsize.values / 24, percentage_undrogued_obs_per_drifter, bins=[100, 100])

# Create a grid of x and y values
x, y = np.meshgrid(x_edges, y_edges)

# Plot the 2D histogram as a heat map
plt.figure(figsize=(10, 6))
plt.pcolormesh(x, y, hist.T, cmap='hot_r', norm=colors.LogNorm())
plt.colorbar(label='Count')
plt.xlabel('trajectory length (days)')
plt.ylabel('undrogued trajectory fraction (%)')

# plt.xscale('log', base=10)

plt.savefig('figures/portion_undrogued_vs_length_heatmap.png', dpi=300)
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


n_traj_gps = ds.location_type.values.sum()
n_obs_gps = ds.rowsize.values[ds.location_type.values].sum()

n_traj_gps_undrogued = np.sum(ds.location_type.values & drifter_lost_drogue)
n_obs_gps_undrogued = ds.rowsize.values[ds.location_type.values & drifter_lost_drogue].sum()

percentage_gps_undrogued_obs = n_obs_gps_undrogued / n_obs_gps * 100
percentage_gps_undrogued_traj = n_traj_gps_undrogued / n_traj_gps * 100

n_obs_gps_undrogued_12km = np.sum(ds.aprox_distance_shoreline.values < 12)
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

