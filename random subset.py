import load_data
import numpy as np
import matplotlib.pyplot as plt
import pickle_manager as pickm
from plotter import *
from analyzer import determine_trapping_event, interpolate_drifter_location
import time
from tqdm import tqdm


def load_random_subset():
    ds = load_data.get_ds_drifters()

    # get the aproximate distance
    df_raster = load_data.get_raster_distance_to_shore_04deg()
    drif_aprox_dist_filename = 'drifter_distances_interpolated_0.04deg_raster'
    ds['aprox_distance_shoreline'] = ('obs', pickm.pickle_wrapper(drif_aprox_dist_filename, interpolate_drifter_location, df_raster, ds))
    n = len(ds.traj)
    traj = np.random.randint(0, len(ds.traj), int(n/100))
    obs = np.where(np.isin(ds.ids, ds.ID[traj]))[0]

    return ds.isel(traj=traj, obs=obs)


ds = pickm.pickle_wrapper('random_subset_2', load_random_subset)

# plot_death_type_bar(ds)
#
# plot_trajectories_death_type(ds)


def tag_drifters_beached(ds, distance_threshold=1000):

    tags = np.zeros(len(ds.traj), dtype=int)

    for i, traj in enumerate(tqdm(ds.traj)):
        if ds.type_death[traj] == 1:
            tags[i] = 1
        else:
            # select a subset of a single trajectory
            obs = np.where(np.isin(ds.ids, ds.ID[traj]))[0]
            ds_i = ds.isel(obs=obs, traj=traj)

            trapping_rows = determine_trapping_event(ds_i.aprox_distance_shoreline.values[-10:],
                                                     np.hypot(ds_i.vn.values[-10:], ds_i.ve.values[-10:]),
                                                     distance_threshold, 0.1)
            if sum(trapping_rows):
                if trapping_rows[-1]:
                    print(f'Found beaching of drifter {ds_i.ID.values}')
                    tags[i] = 1
                elif min(ds_i.aprox_distance_shoreline.values) < distance_threshold:
                    tags[i] = 2

    return tags


thresholds = np.arange(1, 11) ** 2 * 1000
TAGS = np.empty((len(thresholds), len(ds.traj)), dtype=int)
probabilities = np.zeros(len(thresholds), dtype=np.float32)
for i, threshold in enumerate(thresholds):
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
plt.show()