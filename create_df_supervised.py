import numpy as np

import picklemanager as pickm
import pandas as pd
from analyzer import *
import load_data
import plotter

ds = pickm.pickle_wrapper('gdp_random_subset_1', load_data.load_random_subset)
shoreline_resolution = 'i'
#%%
def get_event_indexes(mask, ds):
    # first determine start and end indexes solely based on the mask
    mask = mask.astype(int)
    start_indexes = np.where(np.diff(mask) == 1)[0] + 1
    end_indexes = np.where(np.diff(mask) == -1)[0] + 1

    if mask[0] and not mask[1]:
        start_indexes = np.insert(start_indexes, 0, 0)
    if mask[-1]:
        end_indexes = np.append(end_indexes, len(mask))

    # check if sequencing events should be merged to one event
    duration_threshold = 6 # hours
    times = ds.time.values
    ids = ds.ids.values
    rows_to_delete = []
    for i_event in range(1, len(start_indexes)):
        duration = (times[start_indexes[i_event]] - times[end_indexes[i_event-1]]) / np.timedelta64(1, 'h')
        if 0 < duration <= duration_threshold and ids[i_event] == ids[i_event-1]:
            rows_to_delete.append(i_event-1)
            start_indexes[i_event] = start_indexes[i_event-1]

    start_indexes = np.delete(start_indexes, rows_to_delete)
    end_indexes = np.delete(end_indexes, rows_to_delete)
    return start_indexes, end_indexes


close_2_shore = ds.aprox_distance_shoreline < 10
event_start_indexes, event_end_indexes = get_event_indexes(close_2_shore, ds)


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


beaching_flags, beaching_obs_list = get_beaching_flags(ds, event_start_indexes, event_end_indexes)


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
    no_near_shore_found_indexes = []

    for i_event, event in enumerate(df_gdp.itertuples()):
        min_lon = event.longitude - 0.15
        if min_lon < -180:
            min_lon += 360
        max_lon = event.longitude + 0.15
        if max_lon > 180:
            max_lon -= 360
        min_lat = event.latitude - 0.15
        max_lat = event.latitude + 0.15

        df_shore_box = df_shore[(df_shore['longitude'] >= min_lon) & (df_shore['longitude'] <= max_lon) &
                          (df_shore['latitude'] >= min_lat) & (df_shore['latitude'] <= max_lat)]

        if not df_shore_box.empty:
            i_nearest_shore_point = None

            for i_shore, shore_point in zip(df_shore_box.index, df_shore_box.geometry):
                distance = event.geometry.distance(shore_point)
                if distance < shortest_distances[i_event]:
                    shortest_distances[i_event] = distance
                    i_nearest_shore_point = i_shore

            distances_north[i_event] = event.geometry.y - df_shore_box.geometry[i_nearest_shore_point].y
            distances_east[i_event] = event.geometry.x - df_shore_box.geometry[i_nearest_shore_point].x
        else:
            no_near_shore_found_indexes.append(i_event)
    if no_near_shore_found_indexes:
        print(f'No near shore found for events : {no_near_shore_found_indexes}')
    return shortest_distances, distances_east, distances_north


shortest_distances, distances_east, distances_north = get_distance_and_direction(ds.isel(obs=event_start_indexes))

#%% Create supervised dataframe
n = len(event_start_indexes)
df = pd.DataFrame(data={'time_start': ds.time[event_start_indexes],
                        'time_end': ds.time[event_end_indexes-1],
                        'latitude_start': ds.latitude[event_start_indexes],
                        'latitude_end': ds.latitude[event_end_indexes-1],
                        'longitude_start': ds.longitude[event_start_indexes],
                        'longitude_end': ds.longitude[event_end_indexes-1],
                        've': ds.ve[event_start_indexes],
                        'vn': ds.vn[event_start_indexes],
                        'nearest shore': shortest_distances,
                        'de': distances_east,
                        'dn': distances_north,
                        'beaching_flags': beaching_flags})
df['time_start'] = pd.to_datetime(df['time_start'])
df.sort_values('time_start', inplace=True)
df.filter(['time_start', 'time_end', 'latitude_start', 'longitude_start'], axis=1).to_csv('data/events.csv',
                                                                                          index_label='ID')

#%% Plotting
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
    beaching_event_obs.append([i for i in range(event_start_indexes[i_b], event_end_indexes[i_b])])

extent_offset = 0.1
for i in range(len(beaching_event_obs)):
    ds_select = ds.isel(obs=beaching_event_obs[i])

    extent=(ds_select['longitude'].min()-extent_offset, ds_select['longitude'].max()+extent_offset,
            ds_select['latitude'].min()-extent_offset,  ds_select['latitude'].max()+extent_offset)

    fig, ax = plotter.get_sophie_subplots(figsize=None, extent=extent)
    plot_beaching_trajectories(ds_select, ax, s=12, ds_beaching_obs=ds.isel(obs=beaching_obs_list[i]))
