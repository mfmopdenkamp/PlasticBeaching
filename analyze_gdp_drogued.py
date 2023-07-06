import load_data
import picklemanager as pickm
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

ds = pickm.load_pickle(pickm.create_pickle_path('gdp_drogue_presence'))
#%%

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

plt.savefig('figures/fraction_undrogued_vs_length_heatmap.png', dpi=300)
plt.show()

#%%

# Plot trajectory length histogram
bins = 300
fig, axs = plt.subplots(figsize=(12, 8))
grid = fig.add_gridspec(3, 3)

# Plot 2D histogram in the top right corner
ax_main = fig.add_subplot(grid[0:2, 1:3])
ax_main.pcolormesh(x, y, hist.T, cmap='hot_r', norm=colors.LogNorm())
ax_main.set_xlabel('trajectory length (days)')
ax_main.set_ylabel('undrogued trajectory fraction (%)')
ax_main.set_title('2D Histogram')

# Plot 1D histogram on the left side
ax_left = fig.add_subplot(grid[0:2, 0])
ax_left.hist(percentage_undrogued_obs_per_drifter, bins=100, orientation='horizontal')
ax_left.set_ylabel('undrogued trajectory fraction (%)')
ax_left.yaxis.tick_right()
ax_left.set_xscale('log')

# Plot 1D histogram at the bottom
ax_bottom = fig.add_subplot(grid[2, 1:3])
ax_bottom.hist(ds.rowsize.values / 24, bins=bins, orientation='vertical')
ax_bottom.set_xlabel('trajectory length (days)')
ax_bottom.set_yscale('log')

# Remove unused subplot
fig.delaxes(fig.add_subplot(grid[2, 0]))

# Adjust the size of the subplots
ax_main.set_position([0.35, 0.35, 0.55, 0.55])  # [left, bottom, width, height]
ax_left.set_position([0.05, 0.35, 0.25, 0.55])
ax_bottom.set_position([0.35, 0.05, 0.55, 0.25])

# Add colorbar for the 2D histogram
cax = fig.add_axes([0.92, 0.35, 0.02, 0.55])  # [left, bottom, width, height]
cbar = plt.colorbar(ax_main.collections[0], cax=cax)
cbar.set_label('Count')

# Save the figure
plt.savefig('figures/combined_figure.png', dpi=300)

# Show the figure
plt.show()

