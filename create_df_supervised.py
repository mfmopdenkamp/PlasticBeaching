import numpy as np

import picklemanager as pickm
import pandas as pd
from analyzer import *
import load_data
import plotter

ds = pickm.pickle_wrapper('gdp_random_subset_1', load_data.load_random_subset)

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
    rows_to_delete = []
    for i_event in range(1, len(start_indexes)):
        duration = (times[start_indexes[i_event]] - times[end_indexes[i_event-1]]) / np.timedelta64(1, 'h')
        if 0 < duration <= duration_threshold:
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
    for i, (i_s, i_e) in enumerate(zip(event_start_indexes, event_end_indexes)):

        distance = np.zeros(i_e - i_s, dtype=int)
        velocity = get_absolute_velocity(ds.isel(obs=slice(i_s, i_e)))
        beaching_rows = determine_beaching_event(distance, velocity, max_distance_m=1, max_velocity_mps=0.01)
        if beaching_rows.sum() > 0:
            beaching_flags[i] = True

    return beaching_flags


beaching_flags = get_beaching_flags(ds, event_start_indexes, event_end_indexes)

#%%
def get_distance_and_direction(ds):
    df_shore = load_data.get_shoreline('i', points_only=True)

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
    distances_east = np.zeros(n, dtype=dtype)
    distances_north = np.zeros(n, dtype=dtype)
    no_near_shore_found_indexes = []

    for i_event, event in enumerate(df_gdp.itertuples()):
        min_lon = event.longitude - 0.13
        if min_lon < -180:
            min_lon += 360
        max_lon = event.longitude + 0.13
        if max_lon > 180:
            max_lon -= 360
        min_lat = event.latitude - 0.13
        max_lat = event.latitude + 0.13

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
df = pd.DataFrame(data={'time': ds.time[event_start_indexes],
                        'latitude': ds.latitude[event_start_indexes],
                        'longitude': ds.longitude[event_start_indexes],
                        've': ds.ve[event_start_indexes],
                        'vn': ds.vn[event_start_indexes],
                        'nearest shore': shortest_distances,
                        'de': distances_east,
                        'dn': distances_north,
                        'beaching_flags': beaching_flags})

#%% Plotting
beaching_obs = []
for i_b in np.where(beaching_flags)[0]:
    beaching_obs.append([i for i in range(event_start_indexes[i_b], event_end_indexes[i_b])])


plt, ax = plotter.get_marc_subplots(extent=(df['longitude'].min()-0.12,
                                              df['longitude'].max()+0.12,
                                              df['latitude'].min()-0.12,
                                              df['latitude'].max()+0.12))

plotter.plot_trajectories(ax, ds.isel(obs=beaching_obs[0]))
