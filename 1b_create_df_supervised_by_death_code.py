from numba import jit
import numpy as np
import picklemanager as pickm


ds = pickm.load_pickle(pickm.create_pickle_path('gdp_drogue_presence_sst'))
ds['velocity'] = np.hypot(ds.ve.values, ds.vn.values)
#%%
@jit(nopython=True)
def process_trajectories(traj_values, type_death_values, drifter_ids, traj_idx, undrogue_presence, velocity_values,
                         vn_values, ve_values, latitude_values, longitude_values, time_values):
    n = len(traj_values)
    segments_velocity = np.empty((n, 25))
    segments_vn = np.empty((n, 25))
    segments_ve = np.empty((n, 25))
    segments_latitude = np.empty((n, 25))
    segments_longitude = np.empty((n, 25))
    segments_time = np.empty(n)
    segments_id = np.empty(n)
    segments_ground_flag = np.zeros(n, dtype=bool)

    count = 0
    for j, death_type in zip(traj_values, type_death_values):
        mask_undrogued = undrogue_presence[slice(traj_idx[j], traj_idx[j + 1])]
        mask_undrogue_12km = mask_undrogued & \
                             (ds.aprox_distance_shoreline.values[slice(traj_idx[j], traj_idx[j + 1])] < 12)
        n_undrogued_12km = mask_undrogue_12km.sum()
        if n_undrogued_12km > 24:
            for i in range((n_undrogued_12km-1) // 24):
                slc = slice(traj_idx[j + 1] - (i+1) * 24 - 1, traj_idx[j + 1] - i * 25)
                segments_velocity[count] = velocity_values[slc]
                segments_vn[count] = vn_values[slc]
                segments_ve[count] = ve_values[slc]
                segments_latitude[count] = latitude_values[slc]
                segments_longitude[count] = longitude_values[slc]
                segments_time[count] = time_values.values[slc][0]
                segments_id[count] = drifter_ids[j]

                if death_type == 1 and i == 0:
                    segments_ground_flag[count] = True

                count += 1


        return segments_velocity, segments_latitude,\
               segments_longitude


traj_idx = np.insert(np.cumsum(ds.rowsize.values), 0, 0)

final_segments_drogueness, final_segments_sst, final_segments_velocity, final_segments_latitude,\
final_segments_longitude, rows_to_drop = process_trajectories(
    ds.traj.values, ds.type_death.values, ds.ID.values, traj_idx, np.invert(ds.drogue_presence.values), ds.sst.values,
    ds.velocity.values, ds.latitude.values, ds.longitude.values, ds.time.values)
