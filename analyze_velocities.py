# This script analyzes the velocities within the GDP hourly dataset
from numba import jit
import load_data
import numpy as np
import toolbox as tb
import picklemanager as pickm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ds = pickm.load_pickle(pickm.create_pickle_path('gdp_drogue_presence_sst'))

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
@jit(nopython=True)
def get_velocity(aprox_distances, ve, vn, distances, delta_km=0.2):
    velocities = []
    for i, distance in enumerate(distances):
        mask = ((distance - delta_km) < aprox_distances) & (aprox_distances < distance)
        velocities.append(np.hypot(ve[mask], vn[mask]))
    return velocities


delta_km = 0.4
max_km = 12
distances = np.arange(delta_km, max_km + delta_km, delta_km)

aprox_distances = ds.aprox_distance_shoreline.values[ds.drogue_presence.values]
ve = ds.ve.values[ds.drogue_presence.values]
vn = ds.vn.values[ds.drogue_presence.values]
velocities_drogued = get_velocity(aprox_distances, ve, vn, distances)

aprox_distances = ds.aprox_distance_shoreline.values[~ds.drogue_presence.values]
ve = ds.ve.values[~ds.drogue_presence.values]
vn = ds.vn.values[~ds.drogue_presence.values]
velocities_undrogued = get_velocity(aprox_distances, ve, vn, distances)

#%% Plot the velocities
# Assuming that 'distances' is a common x-axis for both plots
x = np.arange(len(distances))

# Set width of the bars
width = 0.35

# Create figure and axes
fig, ax = plt.subplots(figsize=(13, 5))

# Create the box plots with positions offset by 'width' from 'x'
bp1 = ax.boxplot(velocities_drogued, whis=False, positions=x - width/2, widths=width, patch_artist=True, boxprops=dict(facecolor='b'), medianprops=dict(color='k'), showfliers=False)
bp2 = ax.boxplot(velocities_undrogued, whis=False, positions=x + width/2, widths=width, patch_artist=True, boxprops=dict(facecolor='r'), medianprops=dict(color='k'), showfliers=False)
# ax.violinplot(velocities_drogued, showmeans=True, showmedians=True)
# ax.violinplot(velocities_undrogued, showmeans=True, showmedians=True)

# Add legend manually with patches
legend_patches = [Patch(facecolor='b', edgecolor='k', label='Drogued'),
                  Patch(facecolor='r', edgecolor='k', label='Undrogued')]
ax.legend(handles=legend_patches)

ax.set_xticks(x + width + delta_km/2)
ax.set_xticklabels([f'{x:.1f}' for x in distances])

plt.grid(axis='x')

ax.set_xlabel('Distance to the shoreline (km)')
ax.set_ylabel('Mean velocity (m/s)')

plt.savefig(f'figures/box_plot_mean_velocity_distance_{delta_km}_{max_km}.png')
plt.show()

#%% Plot a histogram of undrogued velocities
fig2, ax = plt.subplots(1,2, figsize=(10, 5), sharey=True, dpi=300)
ax[0].hist(velocities_undrogued[-1], bins=50, color='r', label='Undrogued')
ax[1].hist(velocities_drogued[-1], bins=50, color='b', label='Drogued')
ax[0].set_xlabel('Velocity (m/s)')
ax[0].set_ylabel('Probability density')
ax[1].set_xlabel('Velocity (m/s)')
ax[0].legend()
ax[1].legend()

fig2.suptitle(f'Velocity histogram from {distances[-2]:.1f} to {distances[-1]:.1f} km from the shoreline')

# set subplot spacing
fig2.subplots_adjust(wspace=0.1)

plt.savefig(f'figures/histogram_velocity_{distances[-2]:.1f}_{distances[-1]:.1f}.png')
plt.show()



