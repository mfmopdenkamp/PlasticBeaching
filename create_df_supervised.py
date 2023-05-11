import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import picklemanager as pickm
import pandas as pd
import analyzer as a
import config
import load_data
import plotter
# %% Settings
plot_things = False

percentage = 5
random_set = 1
gps_only = True
undrogued_only = True
threshold_aprox_distance_km = 12

name = f'subset_{percentage}_{(random_set if percentage < 100 else "")}{("_gps" if gps_only else "")}' \
       f'{("_undrogued" if undrogued_only else "")}' \
       f'{("_" + str(threshold_aprox_distance_km) + "km" if threshold_aprox_distance_km is not None else "")}'

ds_gdp = pickm.pickle_wrapper('ds_gdp_' + name, load_data.load_subset, percentage, gps_only, undrogued_only,
                              threshold_aprox_distance_km)
shoreline_resolution = config.shoreline_resolution


# %% Get segment indexes from close2shore mask
def get_segments(ds, seg_len_h):

    s_obs = []
    e_obs = []

    for ID in ds.ID.values:
        obs = ds.obs.values[ds.ids == ID]
        times = ds.time.values[obs]
        durations = (times[1:] - times[:-1]) / np.timedelta64(1, 'h')
        hours_counter = 0
        for ob, duration in zip(obs[1:], durations):
            if hours_counter == 0:
                s_obs.append(obs[0])
            hours_counter += duration
            if hours_counter >= seg_len_h:
                e_obs.append(ob)
                hours_counter = 0
        if hours_counter > 0:
            e_obs.append(obs[-1])

    return np.array(s_obs), np.array(e_obs)


segment_length_h = 24
start_obs, end_obs = get_segments(ds_gdp, segment_length_h)

print('Number of segments before splitting:', len(start_obs))


# %% Get beaching flags and beaching observations
def get_beaching_flags(ds, s_obs, e_obs):
    if len(s_obs) != len(e_obs):
        raise ValueError('"starts_obs" and "end_obs" must have equal lengths!')

    b_flags = np.zeros(len(s_obs), dtype=bool)
    b_obs_list = {}
    for index, (i_s, i_e) in enumerate(zip(s_obs, e_obs)):

        obs = np.arange(i_s, i_e)
        traj = a.traj_from_obs(ds, obs)

        ds_segment = ds.isel(obs=obs, traj=traj)
        mask_drifter_on_shore = a.get_obs_drifter_on_shore(ds_segment)
        if mask_drifter_on_shore.sum() > 0:
            b_flags[index] = True
            b_obs_list[i_s] = mask_drifter_on_shore
    return b_flags, b_obs_list


beaching_flags, beaching_obs_list = get_beaching_flags(ds_gdp, start_obs, end_obs)

print('Number of beaching events:', beaching_flags.sum())

# %% Delete segments that start with beaching observations


mask_start_beaching = np.zeros(len(start_obs), dtype=bool)


for i, start_ob in zip(np.where(beaching_flags)[0], start_obs[beaching_flags]):
    if beaching_obs_list[start_ob][0]:
        mask_start_beaching[i] = True


start_obs = start_obs[~mask_start_beaching]
end_obs = end_obs[~mask_start_beaching]
beaching_flags = beaching_flags[~mask_start_beaching]

print('Number of deleted segments because they start with beaching: ', mask_start_beaching.sum())
print('Number of beaching events:', beaching_flags.sum())


# %% Get shore parameters
def get_shore_parameters(ds):
    df_shore = load_data.get_shoreline(shoreline_resolution, points_only=True)

    df_gdp = a.ds2geopandas_dataframe(ds.latitude.values, ds.longitude.values, df_shore)

    n = len(ds.obs)
    output_dtype = np.float32
    init_distance = np.finfo(output_dtype).max
    shortest_distances = np.ones(n, dtype=output_dtype) * init_distance
    distances_east = np.ones(n, dtype=output_dtype) * init_distance
    distances_north = np.ones(n, dtype=output_dtype) * init_distance
    rmsd = np.zeros(n, dtype=output_dtype)
    no_near_shore_found_indexes = []

    for i_coord, coord in enumerate(df_gdp.itertuples()):

        # get shore points in a box around the coordinate
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

        # determine distance to nearest shore point
        if not df_shore_box.empty:

            distances_shore_box = np.zeros(len(df_shore_box), dtype=output_dtype)
            for i_s, shore_point in enumerate(df_shore_box.geometry):
                distances_shore_box[i_s] = coord.geometry.distance(shore_point)

            shortest_distances[i_coord], i_nearest_shore_point = np.min(distances_shore_box),\
                                                                 np.argmin(distances_shore_box)

            index_point = df_shore_box.index[i_nearest_shore_point]
            # determine direction to this point
            distances_north[i_coord] = coord.geometry.y - df_shore_box.geometry[index_point].y
            distances_east[i_coord] = coord.geometry.x - df_shore_box.geometry[index_point].x

            # fit linear shoreline direction
            x = np.array(df_shore_box.geometry.x.values).reshape(-1, 1)
            y = np.array(df_shore_box.geometry.y.values)

            lr = LinearRegression()
            lr.fit(x, y)
            rmsd[i_coord] = np.sqrt(mean_squared_error(y, lr.predict(x)))

        else:
            no_near_shore_found_indexes.append(i_coord)
    if no_near_shore_found_indexes:
        print(f'No near shore found for segments : {no_near_shore_found_indexes}')
    return shortest_distances, distances_east, distances_north, rmsd, no_near_shore_found_indexes


shortest_distances, distances_east, distances_north, rmsd, no_near_shore_found_indexes = get_shore_parameters(
    ds_gdp.isel(obs=start_obs))

# %% Drop segments where no shore was found (for some reason!!!)
start_obs = np.delete(start_obs, no_near_shore_found_indexes)
end_obs = np.delete(end_obs, no_near_shore_found_indexes)
shortest_distances = np.delete(shortest_distances, no_near_shore_found_indexes)
distances_east = np.delete(distances_east, no_near_shore_found_indexes)
distances_north = np.delete(distances_north, no_near_shore_found_indexes)
beaching_flags = np.delete(beaching_flags, no_near_shore_found_indexes)
rmsd = np.delete(rmsd, no_near_shore_found_indexes)

print('Number of deleted segments because no shore was found: ', len(no_near_shore_found_indexes))
print('Remaining number of beaching events:', beaching_flags.sum())

# %% Create supervised dataframe
n = len(start_obs)
df = pd.DataFrame(data={'time_start': ds_gdp.time[start_obs],
                        'time_end': ds_gdp.time[end_obs - 1],
                        'latitude_start': ds_gdp.latitude[start_obs],
                        'latitude_end': ds_gdp.latitude[end_obs - 1],
                        'longitude_start': ds_gdp.longitude[start_obs],
                        'longitude_end': ds_gdp.longitude[end_obs - 1],
                        've': ds_gdp.ve[start_obs],
                        'vn': ds_gdp.vn[start_obs],
                        'distance_nearest_shore': shortest_distances,
                        'de': distances_east,
                        'dn': distances_north,
                        'beaching_flag': beaching_flags})
# Make sure the time is in datetime format
df['time_start'] = pd.to_datetime(df['time_start'])
df['time_end'] = pd.to_datetime(df['time_end'])

# sort the segments dataframe by time
df.sort_values('time_start', inplace=True)

# %% Save the dataframe to csv

df.to_csv(f'data/segments_{name}.csv', index_label='ID')


# %% Plotting
if plot_things:
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

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
        ds_select = ds_gdp.isel(obs=beaching_event_obs[i])

        extent = (ds_select['longitude'].min() - extent_offset, ds_select['longitude'].max() + extent_offset,
                  ds_select['latitude'].min() - extent_offset, ds_select['latitude'].max() + extent_offset)

        fig, ax = plotter.get_sophie_subplots(figsize=None, extent=extent)
        plot_beaching_trajectories(ds_select, ax, s=12, ds_beaching_obs=ds_gdp.isel(obs=beaching_obs_list[i]))
