from numba import jit
import numpy as np
import picklemanager as pickm


ds = pickm.load_pickle(pickm.create_pickle_path('gdp_drogue_presence'))
hours_to_forecast = 24
#%%
@jit(nopython=True)
def process_trajectories(trajs, type_deaths, drifter_ids, traj_idx, undrogue_presence, vn, ve, latitudes, longitudes,
                         times, aprox_distance_shoreline):
    n = len(undrogue_presence//hours_to_forecast)

    state_vn = np.empty(n)
    state_ve = np.empty(n)
    state_latitude = np.empty(n)
    state_longitude = np.empty(n)
    state_time = np.empty(n)
    state_id = np.empty(n)
    aprox_dist = np.empty(n)
    state_ground_flag = np.zeros(n, dtype=np.bool_)

    count = 0
    for j, death_type, drifter_id in zip(trajs, type_deaths, drifter_ids):
        slice_sel = slice(traj_idx[j], traj_idx[j + 1])
        mask_undrogued = undrogue_presence[slice_sel]
        mask_undrogued_near_shore = mask_undrogued & (aprox_distance_shoreline[slice_sel] < 12)
        n_undrogued_near_shore = mask_undrogued_near_shore.sum()

        vn_sel = vn[slice_sel]
        ve_sel = ve[slice_sel]
        latitude_sel = latitudes[slice_sel]
        longitude_sel = longitudes[slice_sel]
        aprox_dist_sel = aprox_distance_shoreline[slice_sel]
        time_sel = times[slice_sel]

        if death_type == 1:
            state_ground_flag[count] = True
            # Make sure the last state is a grounding state. The trajectory is split by distance to the shoreline, which
            # may split the trajectory in segments that are not connected.
            dts = time_sel[1:] - time_sel[:-1]
            index_from_last = np.arange(np.nonzero(dts[::-1] != 1)[0][-1]+1, n_undrogued_near_shore, hours_to_forecast)
        else:
            index_from_last = np.arange(1, n_undrogued_near_shore, hours_to_forecast)

        for i in index_from_last:
            state_vn[count] = vn_sel[-i]
            state_ve[count] = ve_sel[-i]
            state_latitude[count] = latitude_sel[-i]
            state_longitude[count] = longitude_sel[-i]
            state_time[count] = time_sel[-i]
            aprox_dist[count] = aprox_dist_sel[-i]
            state_id[count] = drifter_id

            count += 1

    return state_vn[:count], state_ve[:count], state_latitude[:count], state_longitude[:count], \
              state_time[:count], state_id[:count], aprox_dist[:count], state_ground_flag[:count]


traj_idx = np.insert(np.cumsum(ds.rowsize.values), 0, 0)

state_vn, state_ve, state_latitude, state_longitude, state_time, state_id, aprox_dist, state_ground_flag = process_trajectories(
    ds.traj.values, ds.type_death.values, ds.ID.values, traj_idx, np.invert(ds.drogue_presence.values), ds.vn.values,
    ds.ve.values, ds.latitude.values, ds.longitude.values, ds.time.values.astype('datetime64[s]').astype('float'),
    ds.aprox_distance_shoreline.values)

#
#%% Write to dataframe
import pandas as pd

state_time = state_time.astype('datetime64[s]')
df = pd.DataFrame(data={'beaching_flag': state_ground_flag,
                        'time': state_time,
                        'drifter_id': state_id,
                        'latitude': state_latitude,
                        'longitude': state_longitude,
                        'velocity_east': state_ve,
                        'velocity_north': state_vn,
                        'speed': np.sqrt(state_vn**2 + state_ve**2),
                        'aprox_distance_shoreline': aprox_dist,
                        })

# Make sure the time is in datetime format
df['time'] = pd.to_datetime(df['time'])

# sort the segments dataframe by time
df.sort_values('time', inplace=True)

# Save the dataframe to csv
df.to_csv(f'data/drifter_state_{hours_to_forecast}.csv', index=False)

