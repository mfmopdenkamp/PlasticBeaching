import numpy as np
from sklearn.linear_model import LinearRegression
import picklemanager as pickm
import pandas as pd
from analyzer import *
import load_data
import plotter
plot_things = False

ds = pickm.pickle_wrapper('gdp_random_subset_1', load_data.load_random_subset)
shoreline_resolution = 'h'
#%%
obs = ds.obs[np.invert(drogue_presence(ds))]
traj = traj_from_obs(ds, obs)
ds = ds.isel(obs=obs, traj=traj)

#%%
def get_event_indexes_from_mask(mask, ds, duration_threshold=6):
    # first determine start and end indexes solely based on the mask
    mask = mask.astype(int)
    start_obs = np.where(np.diff(mask) == 1)[0] + 1
    end_obs = np.where(np.diff(mask) == -1)[0] + 1

    if mask[0] and not mask[1]:
        start_obs = np.insert(start_obs, 0, 0)
    if mask[-1]:
        end_obs = np.append(end_obs, len(mask))

    # check if sequencing events should be merged to one event
     # hours
    event_indexes_to_delete = []

    times = ds.time.values
    ids = ds.ids.values
    for i_event in range(1, len(start_obs)):
        duration = (times[start_obs[i_event]] - times[end_obs[i_event-1]]) / np.timedelta64(1, 'h')
        if 0 < duration <= duration_threshold and ids[i_event] == ids[i_event-1]:
            event_indexes_to_delete.append(i_event-1)
            start_obs[i_event] = start_obs[i_event-1]

    start_obs = np.delete(start_obs, event_indexes_to_delete)
    end_obs = np.delete(end_obs, event_indexes_to_delete)

    return start_obs, end_obs


close_2_shore = ds.aprox_distance_shoreline < 10
event_start_obs, event_end_obs = get_event_indexes_from_mask(close_2_shore, ds)


#%%
def get_beaching_flags(ds, event_start_indexes, event_end_indexes):
    if len(event_start_indexes) != len(event_end_indexes):
        raise ValueError('"event starts" and "event ends" must have equal lengths!')

    beaching_flags = np.zeros(len(event_start_indexes), dtype=bool)
    beaching_obs_list = []
    for i, (i_s, i_e) in enumerate(zip(event_start_indexes, event_end_indexes)):

        # hardcode distance to be neglected by setting distances to zero and threshold to >0 (e.g. 1)
        distance = np.zeros(i_e - i_s, dtype=int)
        velocity = get_absolute_velocity(ds.isel(obs=slice(i_s, i_e)))
        beaching_tags = determine_beaching_obs(distance, velocity, max_distance_m=1, max_velocity_mps=0.01)
        if beaching_tags.sum() > 0:
            beaching_flags[i] = True
            beaching_obs_list.append(np.arange(i_s, i_e)[beaching_tags])
    return beaching_flags, beaching_obs_list


beaching_flags, beaching_obs_list = get_beaching_flags(ds, event_start_obs, event_end_obs)


#%%
def split_events(start_obs, end_obs, beaching_flags, beaching_obs_list, length_threshold=24):
    # Split events based on time. Use index for this, since they correspond to exactly 1 hour.
    # New events may not be smaller than the length threshold!
    # NB: What if split is splitting beaching event?
    split_obs_to_insert = np.array([], dtype=int)
    where_to_insert_event_indexes = np.array([], dtype=int)

    event_lengths = end_obs - start_obs
    split_length = int(length_threshold / 2)
    i_beaching_event = 0
    for i_event, (start_ob, end_ob, event_length, beaching_flag) in enumerate(zip(start_obs, end_obs,
                                                                                  event_lengths, beaching_flags)):

        # determine split points of events based on their length
        if event_length >= length_threshold:

            # if beaching took place, determine consecutive zeros which might split
            if beaching_flag:
                no_beach_count = 0
                event_split_obs = np.array([], dtype=int)

                beach_encountered = False
                beaching_flags_to_insert = np.array([], dtype=bool)
                for ob in range(start_ob, end_ob):

                    if ob not in beaching_obs_list[i_beaching_event]:
                        no_beach_count += 1

                        if no_beach_count >= length_threshold:
                            # add split event
                            event_split_obs = np.append(event_split_obs, ob - split_length + 1)
                            beaching_flags_to_insert = np.append(beaching_flags_to_insert, beach_encountered)

                            beach_encountered = False
                            no_beach_count -= split_length

                    else:
                        no_beach_count = 0
                        beach_encountered = True

                i_beaching_event += 1

            else:
                event_split_obs = np.arange(split_length, event_length - split_length + 1, split_length) \
                                          + start_ob  # start counting from start event instead of zero

            split_obs_to_insert = np.append(split_obs_to_insert, event_split_obs)
            where_to_insert_event_indexes = np.append(where_to_insert_event_indexes,
                                                      np.ones(len(event_split_obs), dtype=int)
                                                      * i_event)

    # insert new events
    start_obs = np.insert(start_obs, where_to_insert_event_indexes + 1, split_obs_to_insert)
    end_obs = np.insert(end_obs, where_to_insert_event_indexes, split_obs_to_insert)

    return start_obs, end_obs


split_events(np.array([0]), np.array([20]), np.array([True]), [[5,6,9,10,11,13,19]], 4)


#%%
def get_distance_and_direction(ds):
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
            i_nearest_shore_point = None

            # determine shortest distance between the subtrajectory and index belonging to this point
            for i_shore, shore_point in zip(df_shore_box.index, df_shore_box.geometry):
                distance = coord.geometry.distance(shore_point)
                if distance < shortest_distances[i_coord]:
                    shortest_distances[i_coord] = distance
                    i_nearest_shore_point = i_shore

            # determine direction to this point
            distances_north[i_coord] = coord.geometry.y - df_shore_box.geometry[i_nearest_shore_point].y
            distances_east[i_coord] = coord.geometry.x - df_shore_box.geometry[i_nearest_shore_point].x

            # determine shoreline direction
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
        print(f'No near shore found for events : {no_near_shore_found_indexes}')
    return shortest_distances, distances_east, distances_north

shortest_distances, distances_east, distances_north = get_distance_and_direction(ds.isel(obs=event_start_obs))

#%% Create supervised dataframe
n = len(event_start_obs)
df = pd.DataFrame(data={'time_start': ds.time[event_start_obs],
                        'time_end': ds.time[event_end_obs - 1],
                        'latitude_start': ds.latitude[event_start_obs],
                        'latitude_end': ds.latitude[event_end_obs - 1],
                        'longitude_start': ds.longitude[event_start_obs],
                        'longitude_end': ds.longitude[event_end_obs - 1],
                        've': ds.ve[event_start_obs],
                        'vn': ds.vn[event_start_obs],
                        'nearest shore': shortest_distances,
                        'de': distances_east,
                        'dn': distances_north,
                        'beaching_flag': beaching_flags})
df['time_start'] = pd.to_datetime(df['time_start'])
df.sort_values('time_start', inplace=True)
df.filter(['time_start', 'time_end', 'latitude_start', 'longitude_start'], axis=1).to_csv('data/event_locations.csv',
                                                                                          index_label='ID')
df.to_csv('data/events_prep.csv', index_label='ID')

#%% Plotting
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
            ax.scatter(ds_beaching_obs.longitude, ds_beaching_obs.latitude, transform=ccrs.PlateCarree(), s=s*2, c='r')

        if not df_shore.empty:
            df_shore.plot(ax=ax, color='b')
        # else:
        #     ax.coastlines()

        plt.tight_layout()
        plt.show()


    beaching_event_obs = []
    for i_b in np.where(beaching_flags)[0]:
        beaching_event_obs.append([i for i in range(event_start_obs[i_b], event_end_obs[i_b])])

    extent_offset = 0.1
    for i in range(len(beaching_event_obs)):
        ds_select = ds.isel(obs=beaching_event_obs[i])

        extent=(ds_select['longitude'].min()-extent_offset, ds_select['longitude'].max()+extent_offset,
                ds_select['latitude'].min()-extent_offset,  ds_select['latitude'].max()+extent_offset)

        fig, ax = plotter.get_sophie_subplots(figsize=None, extent=extent)
        plot_beaching_trajectories(ds_select, ax, s=12, ds_beaching_obs=ds.isel(obs=beaching_obs_list[i]))
