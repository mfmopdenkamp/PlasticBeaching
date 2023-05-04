import numpy as np
from sklearn.linear_model import LinearRegression
import picklemanager as pickm
import pandas as pd
from analyzer import *
import load_data
import plotter


# %% Settings
plot_things = False

percentage = 5
random_set = 2
gps_only = False
undrogued_only = True

name = f'random_subset_{percentage}_{random_set}{("_gps_only" if gps_only else "")}' \
       f'{("_undrogued_only" if undrogued_only else "")}'

ds = pickm.pickle_wrapper('ds_gdp_' + name, load_data.load_random_subset, percentage, gps_only)
shoreline_resolution = 'h'

threshold_duration_hours = 12
threshold_approximate_distance_km = 12
threshold_split_length_h = 24


# %% Get subtraj indexes from close2shore mask
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


close_2_shore = ds.aprox_distance_shoreline.values < threshold_approximate_distance_km
start_obs, end_obs = get_subtraj_indexes_from_mask(close_2_shore, ds, duration_threshold_h=threshold_duration_hours)

print('Number of subtrajs before splitting:', len(start_obs))

# %% Get beaching flags and beaching observations
def get_beaching_flags(ds, start_obs, end_obs):
    if len(start_obs) != len(end_obs):
        raise ValueError('"subtraj starts" and "subtraj ends" must have equal lengths!')

    beaching_flags = np.zeros(len(start_obs), dtype=bool)
    beaching_obs_list = []
    for i, (i_s, i_e) in enumerate(zip(start_obs, end_obs)):

        obs = np.arange(i_s, i_e)
        traj = traj_from_obs(ds, obs)

        ds_subtraj = ds.isel(obs=obs, traj=traj)
        mask_drifter_on_shore = get_obs_drifter_on_shore(ds_subtraj)
        if mask_drifter_on_shore.sum() > 0:
            beaching_flags[i] = True
            beaching_obs_list.append(np.arange(i_s, i_e)[mask_drifter_on_shore])
    return beaching_flags, beaching_obs_list


beaching_flags, beaching_obs_list = get_beaching_flags(ds, start_obs, end_obs)

print('Number of beaching events:', beaching_flags.sum())


# %% Split subtrajectories
def split_subtrajs(start_obs, end_obs, beaching_flags, beaching_obs_list, split_length_h=24):
    # Split subtrajs based on time. Use index for this, since they correspond to exactly 1 hour.
    # New subtrajs may not be smaller than the length threshold!
    split_obs_to_insert = np.array([], dtype=int)
    beaching_flags_to_insert = np.array([], dtype=bool)
    where_to_insert_new_subtraj = np.array([], dtype=int)
    where_to_change_beaching_flags = np.array([], dtype=int)
    subtraj_lengths = end_obs - start_obs

    i_beaching_event = 0

    for i_subtraj, (start_ob, end_ob, subtraj_length, beaching_flag) in enumerate(zip(start_obs, end_obs,
                                                                                      subtraj_lengths, beaching_flags)):

        # determine split points of subtrajs based on their length
        if subtraj_length >= split_length_h * 2:
            subtraj_split_obs = np.arange(split_length_h, subtraj_length - split_length_h + 1, split_length_h) \
                                + start_ob  # start counting from start subtraj instead of zero

            # if beaching took place, check new beaching flags
            if beaching_flag:

                beaching_obs = beaching_obs_list[i_beaching_event]

                # check if original beaching flag must be changed to False
                if not np.any(np.isin(np.arange(start_ob, subtraj_split_obs[0]), beaching_obs)):
                    where_to_change_beaching_flags = np.append(where_to_change_beaching_flags, i_subtraj)

                new_beaching_flags_from_subtraj = np.array(np.zeros(len(subtraj_split_obs)), dtype=bool)
                for j in range(len(subtraj_split_obs) - 1):
                    obs_in_new_sub_traj = np.arange(subtraj_split_obs[j], subtraj_split_obs[j+1])
                    if np.any(np.isin(obs_in_new_sub_traj, beaching_obs)):
                        new_beaching_flags_from_subtraj[j] = True

                # check if the last part of the subtraj is beaching
                if np.any(np.isin(np.arange(subtraj_split_obs[-1], end_ob), beaching_obs)):
                    new_beaching_flags_from_subtraj[-1] = True

                if sum(new_beaching_flags_from_subtraj) < 1:
                    raise ValueError('No beaching flag was set for subtraj', i_subtraj)

            # if no beaching took place, set all new beaching flags to False
            else:
                new_beaching_flags_from_subtraj = np.zeros(len(subtraj_split_obs), dtype=bool)

            # append new split points for this coordinate
            split_obs_to_insert = np.append(split_obs_to_insert, subtraj_split_obs)
            beaching_flags_to_insert = np.append(beaching_flags_to_insert, new_beaching_flags_from_subtraj)
            where_to_insert_new_subtraj = np.append(where_to_insert_new_subtraj,
                                                    np.ones(len(subtraj_split_obs), dtype=int) * i_subtraj)

        if beaching_flag:
            i_beaching_event += 1
    # change beaching flags
    beaching_flags[where_to_change_beaching_flags] = False

    # insert new subtrajs
    start_obs = np.insert(start_obs, where_to_insert_new_subtraj + 1, split_obs_to_insert)
    end_obs = np.insert(end_obs, where_to_insert_new_subtraj, split_obs_to_insert)
    beaching_flags = np.insert(beaching_flags, where_to_insert_new_subtraj + 1, beaching_flags_to_insert)

    return start_obs, end_obs, beaching_flags


start_obs, end_obs, beaching_flags = split_subtrajs(start_obs, end_obs, beaching_flags, beaching_obs_list,
                                                    split_length_h=threshold_split_length_h)

print('Number of subtrajs after splitting: ', len(start_obs))
print('Number of beaching events:', beaching_flags.sum())

# %% Delete subtrajs that start with beaching observations

all_beaching_obs = np.concatenate(beaching_obs_list)
mask_start_beaching = np.in1d(start_obs, all_beaching_obs)
start_obs = start_obs[~mask_start_beaching]
end_obs = end_obs[~mask_start_beaching]
beaching_flags = beaching_flags[~mask_start_beaching]

print('Number of deleted subtrajs because they start with beaching: ', mask_start_beaching.sum())
print('Number of beaching events:', beaching_flags.sum())


# %% Get shore parameters
def get_shore_parameters(ds):
    df_shore = load_data.get_shoreline(shoreline_resolution, points_only=True)

    lats = ds.latitude.values
    lons = ds.longitude.values
    df_gdp = gpd.GeoDataFrame({'latitude': lats, 'longitude': lons},
                              geometry=gpd.points_from_xy(lons, lats),
                              crs='epsg:4326')

    # Make sure the CRS is identical.
    df_gdp.to_crs(df_shore.crs, inplace=True)

    n = len(ds.obs)
    dtype = np.float32
    init_distance = np.finfo(dtype).max
    shortest_distances = np.ones(n, dtype=dtype) * init_distance
    distances_east = np.ones(n, dtype=dtype) * init_distance
    distances_north = np.ones(n, dtype=dtype) * init_distance
    shoreline_angles = np.ones(n, dtype=np.float16) * 9
    no_near_shore_found_indexes = []

    for i_coord, coord in enumerate(df_gdp.itertuples()):
        min_lon = coord.longitude - 0.15
        if min_lon < -180:
            min_lon += 360
        max_lon = coord.longitude + 0.15
        if max_lon > 180:
            max_lon -= 360
        min_lat = coord.latitude - 0.15
        max_lat = coord.latitude + 0.15

        df_shore_box = df_shore[(df_shore['longitude'] >= min_lon) & (df_shore['longitude'] <= max_lon) &
                                (df_shore['latitude'] >= min_lat) & (df_shore['latitude'] <= max_lat)]

        if not df_shore_box.empty:

            # determine shortest distance between the subtrajectory and index belonging to this point
            distances_shore_box = np.zeros(len(df_shore_box), dtype=dtype)
            for i_s, shore_point in enumerate(df_shore_box.geometry):
                distance = coord.geometry.distance(shore_point)
                distances_shore_box[i_s] = distance

            shortest_distances[i_coord], i_nearest_shore_point = distances_shore_box.min(), distances_shore_box.argmin()

            index_point = df_shore_box.index[i_nearest_shore_point]
            # determine direction to this point
            distances_north[i_coord] = coord.geometry.y - df_shore_box.geometry[index_point].y
            distances_east[i_coord] = coord.geometry.x - df_shore_box.geometry[index_point].x

            # fit linear shoreline direction
            x = np.array(df_shore_box.geometry.x.values).reshape(-1, 1)
            y = np.array(df_shore_box.geometry.y.values)

            lr = LinearRegression()
            lr.fit(x, y)
            intercept = lr.intercept_
            slope = lr.coef_[0]
            angle = np.arctan(slope)
            shoreline_angles[i_coord] = angle

            # df_shore_box.plot()
            # plt.annotate(f'slope = {slope:.2f}\nangle = {angle:.2f}', xy=(0.9, 0.9), xycoords='axes fraction')
            # plt.plot(x.reshape(-1), np.polyval([slope, intercept], x.reshape(-1)))
            # plt.scatter(coord.geometry.x, coord.geometry.y, s=10, c='r')
            # plt.show()

        else:
            no_near_shore_found_indexes.append(i_coord)
    if no_near_shore_found_indexes:
        print(f'No near shore found for subtrajs : {no_near_shore_found_indexes}')
    return shortest_distances, distances_east, distances_north, no_near_shore_found_indexes


shortest_distances, distances_east, distances_north, no_near_shore_found_indexes = get_shore_parameters(
    ds.isel(obs=start_obs))

# %% Drop subtrajs where no shore was found (for some reason!!!)
start_obs = np.delete(start_obs, no_near_shore_found_indexes)
end_obs = np.delete(end_obs, no_near_shore_found_indexes)
shortest_distances = np.delete(shortest_distances, no_near_shore_found_indexes)
distances_east = np.delete(distances_east, no_near_shore_found_indexes)
distances_north = np.delete(distances_north, no_near_shore_found_indexes)
beaching_flags = np.delete(beaching_flags, no_near_shore_found_indexes)

print('Number of deleted subtrajs because no shore was found: ', len(no_near_shore_found_indexes))
print('Remaining number of beaching events:', beaching_flags.sum())

# %% Create supervised dataframe
n = len(start_obs)
df = pd.DataFrame(data={'time_start': ds.time[start_obs],
                        'time_end': ds.time[end_obs - 1],
                        'latitude_start': ds.latitude[start_obs],
                        'latitude_end': ds.latitude[end_obs - 1],
                        'longitude_start': ds.longitude[start_obs],
                        'longitude_end': ds.longitude[end_obs - 1],
                        've': ds.ve[start_obs],
                        'vn': ds.vn[start_obs],
                        'distance_nearest_shore': shortest_distances,
                        'de': distances_east,
                        'dn': distances_north,
                        'beaching_flag': beaching_flags})
# Make sure the time is in datetime format
df['time_start'] = pd.to_datetime(df['time_start'])
df['time_end'] = pd.to_datetime(df['time_end'])

# sort the subtraj dataframe by time
df.sort_values('time_start', inplace=True)

# %% Save the dataframe to csv

df.to_csv(f'data/subtrajs_{name}.csv', index_label='ID')


# %% Plotting
if plot_things:
    import cartopy.crs as ccrs

    def plot_beaching_trajectories(ds, ax=None, s=15, ds_beaching_obs=None, df_shore=pd.DataFrame()):
        """given a dataset, plot the trajectories on a map"""
        if ax is None:
            plt.figure(figsize=(12, 8), dpi=300)
            ax = plt.axes(projection=ccrs.PlateCarree())
            extent_offset = 0.2
            ax.set_xlim([ds['longitude'].min() - extent_offset, ds['longitude'].max() + extent_offset])
            ax.set_ylim([ds['latitude'].min() - extent_offset, ds['latitude'].max() + extent_offset])

        ax.scatter(ds.longitude, ds.latitude, transform=ccrs.PlateCarree(), s=s, c='midnightblue', alpha=0.5)
        ax.plot(ds.longitude, ds.latitude, ':k', transform=ccrs.PlateCarree(), alpha=0.5)

        if ds_beaching_obs is not None:
            ax.scatter(ds_beaching_obs.longitude, ds_beaching_obs.latitude, transform=ccrs.PlateCarree(), s=s * 2,
                       c='r')

        if not df_shore.empty:
            df_shore.plot(ax=ax, color='b')
        # else:
        #     ax.coastlines()

        plt.tight_layout()
        plt.show()


    beaching_event_obs = []
    for i_b in np.where(beaching_flags)[0]:
        beaching_event_obs.append([i for i in range(start_obs[i_b], end_obs[i_b])])

    extent_offset = 0.1
    for i in range(len(beaching_event_obs)):
        ds_select = ds.isel(obs=beaching_event_obs[i])

        extent = (ds_select['longitude'].min() - extent_offset, ds_select['longitude'].max() + extent_offset,
                  ds_select['latitude'].min() - extent_offset, ds_select['latitude'].max() + extent_offset)

        fig, ax = plotter.get_sophie_subplots(figsize=None, extent=extent)
        plot_beaching_trajectories(ds_select, ax, s=12, ds_beaching_obs=ds.isel(obs=beaching_obs_list[i]))
