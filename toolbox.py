import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import math
from tqdm import tqdm
import pandas as pd


colors = [
        (0, 0, 0),              # black
        (0, 0.147, 0.698),    # blue
        (0.902, 0.624, 0),  # gold
        (0.875, 0.088, 0.012),  # red
        (0.002, 0.586, 0.09),   # green
        (0.016, 0.494, 0.822),  # sky blue
        (0.596, 0.306, 0.639),  # purple
        (1, 0.498, 0.3),          # orange
        (0.659, 0.439, 0.204),  # brown
        (0.337, 0.337, 0.337),  # gray
    ]


markers = ['o', 'v', 's', 'D', 'P', 'X', 'd', 'p', 'h', '8']


def end_date_2_years(ds_end_date):
    return ds_end_date.values.astype('datetime64[Y]').astype(int) + 1970


def find_index_last_obs_traj(ds):
    return np.append(np.where(np.diff(ds.ids.values))[0], -1)


def obs_from_traj(ds, traj):
    return np.where(np.isin(ds.ids.values, ds.ID.values[traj]))[0]


def traj_from_obs(ds, obs):
    return np.where(np.isin(ds.ID.values, np.unique(ds.ids.values[obs])))[0]


def get_absolute_velocity(ds):
    return np.hypot(ds.vn.values, ds.ve.values)


def dataset2geopandas(lats, lons, df_shore):

    gdf = gpd.GeoDataFrame({'latitude': lats, 'longitude': lons},
                           geometry=gpd.points_from_xy(lons, lats),
                           crs='epsg:4326')
    gdf.to_crs(df_shore.crs, inplace=True)
    return gdf


def find_shortest_distance(ds_gdp, gdf_shoreline):
    """"Calculates distance from shoreline (Polygons) to drifters for every drifters location (Point)"""
    lats = ds_gdp.latitude.values
    lons = ds_gdp.longitude.values
    # create geopandas dataframe from xarray dataset
    gdf_gdp = gpd.GeoDataFrame({'latitude': lats, 'longitude': lons},
                              geometry=gpd.points_from_xy(lons, lats),
                              crs='epsg:4326')

    gdf_shoreline.to_crs(crs=3857, inplace=True)
    gdf_gdp.to_crs(gdf_shoreline.crs, inplace=True)

    dtype = np.float32
    init_distance = np.finfo(dtype).max
    shortest_distances = np.ones(len(gdf_gdp), dtype=dtype) * init_distance

    for i, point in enumerate(tqdm(gdf_gdp.geometry)):
        for polygon in gdf_shoreline.geometry:
            distance = point.distance(polygon)
            if distance < shortest_distances[i]:
                shortest_distances[i] = distance

    return shortest_distances


def haversine_distance(lat1, lon1, lat2, lon2):
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of earth in meters
    radius_m = 6371 * 1000

    # calculate the result
    distance = radius_m * c
    return distance


def longitude_translator(lon):
    if isinstance(lon, int) or isinstance(lon, float):
        if lon < 0:
            return 360 + lon
        else:
            return lon
    elif isinstance(lon, np.ndarray):
        lon[lon < 0] += 360
        return lon

def get_lonlatbox(lon, lat, side_length):
    """side_length in meters"""
    lon_length = side_length / (111320 * math.cos(math.radians(lat))) / 2  # 1 degree longitude is 111320m * cos(latitude)
    min_lon = lon - lon_length
    if min_lon < -180:
        min_lon += 360
    max_lon = lon + lon_length
    if max_lon > 180:
        max_lon -= 360

    lat_length = side_length / 111320 / 2  # 1 degree latitude is 111.32 km
    min_lat = lat - lat_length
    max_lat = lat + lat_length

    return min_lon, max_lon, min_lat, max_lat


def get_obs_drifter_on_shore(ds, minimum_time_h=4):
    """
    Determine beaching events based on two thresholds. The distance travelled between observations and distance between
    beginning and end of a threshold number of observations.
    :param ds:  xarray dataset containing the drifter data
    :param threshold_distance_start_end_m:  distance threshold between beginning and end of minimum_time_h in meters
    :param threshold_distance_travelled_m:  travelled distance threshold in meters
    :param minimum_time_h:  number of hours that the drifter must be within the threshold distances
    :return:  boolean array with True for drifter on shore
    """

    # check if minimum_time_h is an integer and greater than 0
    if not isinstance(minimum_time_h, int) or minimum_time_h < 0:
        raise ValueError('minimum_time_h must be an integer greater than 0')

    # check if only a single drifter is given as input otherwise raise error
    if len(np.unique(ds.ids)) != 1:
        raise ValueError('Only a single trajectory can be given as input')

    beaching_tags = np.zeros(len(ds.obs), dtype=bool)

    # check if there are enough observations
    if len(ds.obs) < minimum_time_h + 1:
        return beaching_tags

    # for Argos look at velocities only
    if not ds.location_type.values:
        velocities = get_absolute_velocity(ds)

        count = 0
        a = np.hypot(ds.err_vn.values, ds.err_ve.values) * 2
        b = np.clip(a, 0, 0.02)
        max_velocity_mps = np.mean(b)
        for i,  v in enumerate(velocities):
            if v < max_velocity_mps:
                count += 1

                if count == minimum_time_h:
                    beaching_tags[i - minimum_time_h:i] = True
                elif count > minimum_time_h:
                    beaching_tags[i] = True

            else:
                count = 0

    # for GPS: check if the preceding observations lay within a threshold radius/distance from the current observation
    else:
        lats = ds.latitude.values
        lons = ds.longitude.values

        df_gdp = gpd.GeoDataFrame({'latitude': lats, 'longitude': lons},
                                  geometry=gpd.points_from_xy(lons, lats),
                                  crs='epsg:4326')
        df_gdp.to_crs(epsg=3857, inplace=True)

        x = df_gdp.geometry.x.values
        y = df_gdp.geometry.y.values

        # compute threshold distances in meters
        err_lons = ds.err_lon.values
        err_lats = ds.err_lat.values
        err_distances_m_0 = haversine_distance(lats, lons, lats + err_lats, lons + err_lons)

        # distances are small so use simple euclidean distance in meters
        # convert error in lat/lon to error in meters
        err_lons_m = err_lons * 111000 * np.cos(np.deg2rad(lats))
        err_lats_m = err_lats * 111000
        err_distances_m = np.hypot(err_lons_m, err_lats_m)

        # this method computes for n-minimum_time_h observations the distances between the observation and
        # minimum_time_h observations before and puts these in a matrix.

        D = np.zeros((minimum_time_h, len(lats) - minimum_time_h))
        for i in range(len(lats) - minimum_time_h):
            from_ob = i + minimum_time_h
            for j in range(minimum_time_h):
                to_ob = from_ob - j - 1
                D[j, i] = np.hypot(x[from_ob] - x[to_ob], y[from_ob] - y[to_ob])

        for i in range(len(lats) - minimum_time_h):
            # arbitrary factor 2 because 1 std would only include 68% of the data
            # if np.all(D[:, i] < err_distances_m[i:i + minimum_time_h] * 2):
            if np.all(D[:, i] < np.clip(err_distances_m[i:i + minimum_time_h], 0, 100)):
                beaching_tags[i:i + minimum_time_h] = True

    return beaching_tags


def days_without_drogue(ds):

    drogue_lost_dates = ds.drogue_lost_date.values
    end_date = ds.end_date.values

    days = []
    for dld, ed in zip(drogue_lost_dates, end_date):
        try:
            days.append((ed - dld) / np.timedelta64(1, 'D'))
        except:
            pass

    days = np.array(days)
    return days


def tag_drifters_beached(ds, distance_threshold=1000):

    tags = np.zeros(len(ds.traj), dtype=int)

    for i, traj in enumerate(ds.traj):
        if ds.type_death[traj] == 1:
            tags[i] = 1
        else:
            # select a subset of a single trajectory
            obs = obs_from_traj(ds, traj)
            ds_i = ds.isel(obs=obs, traj=traj)

            beaching_rows = get_obs_drifter_on_shore(ds_i.aprox_distance_shoreline.values[-10:],
                                                     np.hypot(ds_i.vn.values[-10:], ds_i.ve.values[-10:]),
                                                     distance_threshold, 0.1)

            if beaching_rows[-1]:
                print(f'Found beaching of drifter {ds_i.ID.values}')
                tags[i] = 1
            elif min(ds_i.aprox_distance_shoreline.values) < distance_threshold:
                tags[i] = 2

    return tags


def probability_distance_sophie(ds, verbose=True):
    thresholds = np.logspace(-1, 6, num=15, base=4)
    TAGS = np.empty((len(thresholds), len(ds.traj)), dtype=int)
    probabilities = np.zeros(len(thresholds), dtype=np.float32)
    for i, threshold in enumerate(tqdm(thresholds)):
        tags = tag_drifters_beached(ds, distance_threshold=threshold)
        TAGS[i, :] = tags

    for i in range(len(thresholds)):
        n_ones = np.sum(TAGS[i, :] == 1)
        n_twos = np.sum(TAGS[i, :] == 2)
        probabilities[i] = n_ones / (n_ones + n_twos)

    if verbose:
        plt.figure()
        plt.plot(thresholds / 1000, probabilities)
        plt.xlabel('distance threshold [km]')
        plt.ylabel('probability to find beaching')
        plt.semilogx()
        plt.show()


def get_drogue_presence(ds, coords=False):
    # only works for full dataset where rowsize is true!

    obs = ds.obs.values
    drogue_lost_dates = ds.drogue_lost_date.values

    ids = ds.ids.values
    drogue_presence = np.ones(len(obs), dtype=bool)
    if coords:
        latitude = ds.latitude.values
        longitude = ds.longitude.values
        lats = np.zeros(len(ds.traj), dtype=np.float32)
        lons = np.zeros(len(ds.traj), dtype=np.float32)

    traj_idx = np.insert(np.cumsum(ds.rowsize.values), 0, 0)
    for i, (drogue_lost_date) in enumerate(tqdm(drogue_lost_dates)):
        # if drogue lost date is not known, assume it is always present
        if not np.isnat(drogue_lost_date):
            obs_id = obs[slice(traj_idx[i], traj_idx[i + 1])]
            times = ds.time.values[obs_id]
            mask = times < drogue_lost_date
            drogue_presence[obs_id] = mask
            if coords:
                obs_change = obs_id[np.argmin(np.diff(mask))] + 1
                lats[i] = latitude[obs_change]
                lons[i] = longitude[obs_change]

    if not coords:
        return drogue_presence
    else:
        return drogue_presence, lats, lons


def get_density_grid(latitude, longitude, xlim=None, ylim=None, latlon_box_size=2, lat_box_size=None, lon_box_size=None):
    # filter lat lons on xlim and ylim
    if xlim is not None:
        xlim_mask = (longitude > xlim[0]) & (longitude < xlim[1])
        latitude = latitude[xlim_mask]
        longitude = longitude[xlim_mask]
    if ylim is not None:
        ylim_mask = (latitude > ylim[0]) & (latitude < ylim[1])
        latitude = latitude[ylim_mask]
        longitude = longitude[ylim_mask]

    df = pd.DataFrame({'lat': latitude, 'lon': longitude})

    if latlon_box_size is not None:
        lat_box_size = latlon_box_size
        lon_box_size = latlon_box_size

    # Calculate the box indices for each coordinate
    df['lat_box'] = ((df['lat'] + 90) // lat_box_size)
    df['lon_box'] = ((df['lon'] + 180) // lon_box_size)

    # reduce lon_box to 0-max_lat_box_id and lat_box to 0-max_lon_box_id for drifters exactly on 90N or 180E
    df['lon_box'] = df['lon_box'] % int(360 / lon_box_size)
    df['lat_box'] = df['lat_box'] % int(180 / lat_box_size)

    # Group the coordinates by box indices and count the number of coordinates in each box
    grouped = df.groupby(['lat_box', 'lon_box']).size().reset_index(name='count')

    # Create an empty grid to store the density values
    density_grid = np.zeros((int(180 / lat_box_size), int(360 / lon_box_size)))

    # Fill the density grid with the count values
    for _, row in grouped.iterrows():
        i_lat_box = int(row['lat_box'])
        i_lon_box = int(row['lon_box'])

        count = row['count']
        density_grid[i_lat_box, i_lon_box] = count

    X, Y = np.meshgrid(np.arange(-180, 180, lon_box_size) + lon_box_size / 2,
                       np.arange(-90, 90, lat_box_size) + lat_box_size / 2)

    return X, Y, density_grid


def normalize_array(arr):
    # Shift the array to have only positive values
    shifted_arr = arr - np.min(arr)

    # Scale the shifted array to the range [0, 1]
    normalized_arr = shifted_arr / np.max(shifted_arr)

    return normalized_arr


def get_probabilities(df, column_names, split_points):
    n = df.shape[0]
    df['all'] = np.zeros(n)
    for column in column_names:
        df['all'] += df[column]
    column_names.insert(0, 'all')

    grounding_prob_smaller = {}
    grounding_prob_larger = {}
    for column in column_names:
        df[column] = normalize_array(df[column])
        grounding_prob_smaller[column] = np.zeros(len(split_points))
        grounding_prob_larger[column] = np.zeros(len(split_points))
        for i, score_threshold in enumerate(split_points):
            df_filtered = df[df[column] <= score_threshold]
            df_filtered_larger = df[df[column] > score_threshold]
            grounding_prob_smaller[column][i] = df_filtered.beaching_flag.mean()
            grounding_prob_larger[column][i] = df_filtered_larger.beaching_flag.mean()

    return grounding_prob_smaller, grounding_prob_larger


def get_impurity(node):
    """
    :param node: data set on which to calculate
    :return: impurity based on gini-index
    """
    n = len(node)
    if n == 0:
        return 0
    else:
        return np.sum(node)/n * (1 - np.sum(node)/n)


def get_impurity_reduction(df, column_names, split_points):
    n = df.shape[0]
    df['all'] = np.zeros(n)
    for column in column_names:
        df['all'] += df[column]
    column_names.insert(0, 'all')

    impurity_reductions = {}
    original_impurity = get_impurity(df.beaching_flag)
    for column in column_names:
        df[column] = normalize_array(df[column])
        impurity_reductions[column] = np.zeros(len(split_points))
        for i, split_point in enumerate(split_points):
            if split_point == 1 or split_point == 0:
                impurity_reductions[column][i] = 0
                continue
            left_child = df.beaching_flag[df[column] < split_point]
            right_child = df.beaching_flag[df[column] > split_point]
            reduction = original_impurity - (len(left_child) / n * get_impurity(left_child) + len(right_child) / n * get_impurity(right_child))
            impurity_reductions[column][i] = reduction

    return impurity_reductions

