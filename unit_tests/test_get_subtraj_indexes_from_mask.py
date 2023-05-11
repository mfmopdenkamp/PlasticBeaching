
# test the function get_subtraj_indexes_from_mask

import numpy as np
import xarray as xr
import pandas as pd
import picklemanager as pickm
import load_data
import analyzer as a


def no_time_check(mask, ds):
    # first determine start and end indexes solely based on the mask
    mask = mask.astype(int)
    obs = ds.obs.values
    start_obs = obs[1:][np.diff(mask) == 1]
    end_obs = obs[1:][np.diff(mask) == -1]

    if mask[0]:
        start_obs = np.insert(start_obs, 0, 0)
    if mask[-1]:
        end_obs = np.append(end_obs, len(mask))

    # split subtrajs that dont belong to single drifter
    ids = ds.ids.values

    obs_where_to_split = obs[1:][np.array(np.diff(ids) & mask[:-1] & mask[1:], dtype=bool)]
    start_obs = np.append(start_obs, obs_where_to_split)
    end_obs = np.append(end_obs, obs_where_to_split)
    start_obs = np.sort(start_obs)
    end_obs = np.sort(end_obs)

    return start_obs, end_obs


# create random boolean mask array
mask = np.array([True, True, False, True, True, True, True, False, True, True])
ids = np.array([2,2,2,2,2,2,3,3,3,3])

# create xarray dataset with 'ids' and 'mask' arrays
ds = xr.Dataset(
    data_vars={
        'ids': (('x'), ids),
    },
    coords={
        'obs': np.arange(10)
    }
)


# test the function
start_obs_test, end_obs_test = no_time_check(mask, ds)


percentage = 5
random_set = 2
gps_only = True
undrogued_only = True
ds = pickm.pickle_wrapper(f'gdp_random_subset_{percentage}_{random_set}'
                          f'{("_gps_only" if gps_only else "")}'
                          f'{("_undrogued_only" if undrogued_only else "")}',
                          load_data.load_subset, percentage, gps_only)


def get_subtraj_indexes_from_mask(mask, ds, duration_threshold_h=12):
    # first determine start and end indexes solely based on the mask
    mask = mask.astype(int)
    obs = ds.obs.values
    start_obs = obs[1:][np.diff(mask) == 1]
    end_obs = obs[1:][np.diff(mask) == -1]

    if mask[0]:
        start_obs = np.insert(start_obs, 0, 0)
    if mask[-1]:
        end_obs = np.append(end_obs, len(mask))

    # split subtrajs that dont belong to single drifter
    ids = ds.ids.values

    obs_where_to_split = obs[1:][np.array(np.diff(ids).astype(bool) & mask[:-1] & mask[1:], dtype=bool)]
    start_obs = np.append(start_obs, obs_where_to_split)
    end_obs = np.append(end_obs, obs_where_to_split)
    start_obs = np.sort(start_obs)
    end_obs = np.sort(end_obs)

    # check if sequencing subtraj should be merged to one subtraj
    subtraj_indexes_to_delete = []

    times = ds.time.values

    for i_sj in range(1, len(start_obs)):
        if ids[i_sj] == ids[i_sj - 1]:
            duration = (times[start_obs[i_sj]] - times[end_obs[i_sj - 1]]) / np.timedelta64(1, 'h')
            if 0 < duration <= duration_threshold_h:
                subtraj_indexes_to_delete.append(i_sj - 1)
                start_obs[i_sj] = start_obs[i_sj - 1]

    start_obs = np.delete(start_obs, subtraj_indexes_to_delete)
    end_obs = np.delete(end_obs, subtraj_indexes_to_delete)

    return start_obs, end_obs


threshold_duration_hours = 12
threshold_approximate_distance_km = 12

obs = np.arange(462538, 462801)
traj = a.traj_from_obs(ds, obs)
ds_subtraj = ds.isel(obs=obs, traj=traj)
close_2_shore = ds_subtraj.aprox_distance_shoreline.values < threshold_approximate_distance_km

df = pd.DataFrame(data={'mask':close_2_shore, 'ids': ds_subtraj.ids.values, 'obs': ds_subtraj.obs.values})

start_obs, end_obs = get_subtraj_indexes_from_mask(close_2_shore, ds_subtraj, threshold_duration_hours)


print('Done.')

