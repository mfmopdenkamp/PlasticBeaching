import load_data
import numpy as np
import matplotlib.pyplot as plt
import picklemanager as pickm
from plotter import *
from analyzer import determine_beaching_event, interpolate_drifter_location
import time
from tqdm import tqdm


def load_random_subset():
    ds = load_data.get_ds_drifters()

    n = len(ds.traj)
    traj = np.random.choice(np.arange(len(ds.traj)), size=int(n/100), replace=False)
    obs = np.where(np.isin(ds.ids, ds.ID[traj]))[0]
    ds_subset = ds.isel(traj=traj, obs=obs)
    df_raster = load_data.get_raster_distance_to_shore_04deg()
    ds_subset['aprox_distance_shoreline'] = ('obs', interpolate_drifter_location(df_raster, ds_subset, method='nearest'))

    return ds_subset


ds = pickm.pickle_wrapper('gdp_random_subset_1', load_random_subset)
# ds = load_data.get_ds_drifters(filename='gdp_v2.00.nc_approx_dist_nearest')

# plot_death_type_bar(ds)
#
# plot_trajectories_death_type(ds)


def tag_drifters_beached(ds, distance_threshold=1000):

    tags = np.zeros(len(ds.traj), dtype=int)

    for i, traj in enumerate(ds.traj):
        if ds.type_death[traj] == 1:
            tags[i] = 1
        else:
            # select a subset of a single trajectory
            obs = np.where(np.isin(ds.ids, ds.ID[traj]))[0]
            ds_i = ds.isel(obs=obs, traj=traj)

            beaching_rows = determine_beaching_event(ds_i.aprox_distance_shoreline.values[-10:],
                                                     np.hypot(ds_i.vn.values[-10:], ds_i.ve.values[-10:]),
                                                     distance_threshold, 0.1)

            if beaching_rows[-1]:
                print(f'Found beaching of drifter {ds_i.ID.values}')
                tags[i] = 1
            elif min(ds_i.aprox_distance_shoreline.values) < distance_threshold:
                tags[i] = 2

    return tags


thresholds = np.logspace(-1, 6, num=15, base=4)
TAGS = np.empty((len(thresholds), len(ds.traj)), dtype=int)
probabilities = np.zeros(len(thresholds), dtype=np.float32)
for i, threshold in enumerate(tqdm(thresholds)):
    tags = tag_drifters_beached(ds, distance_threshold=threshold)
    TAGS[i, :] = tags

for i in range(len(thresholds)):
    n_ones = np.count_nonzero(TAGS[i, :] == 1)
    n_twos = np.count_nonzero(TAGS[i, :] == 2)
    probabilities[i] = n_ones / (n_ones+n_twos)

plt.figure()
plt.plot(thresholds / 1000, probabilities)
plt.xlabel('distance threshold [km]')
plt.ylabel('probability to find beaching')
plt.semilogx()
plt.show()