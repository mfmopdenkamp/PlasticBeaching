import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import pandas as pd
from tqdm import tqdm
import time
from scipy.interpolate import griddata


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


def ds2geopandas_dataframe(lats, lons, df_shore):

    gdf = gpd.GeoDataFrame({'latitude': lats, 'longitude': lons},
                           geometry=gpd.points_from_xy(lons, lats),
                           crs='epsg:4326')
    gdf.to_crs(df_shore.crs, inplace=True)
    return gdf


def interpolate_drifter_location(df_raster, ds_drifter, method='linear'):
    # Drifter locations (longitude and latitude)
    drifter_lon = ds_drifter.longitude.values
    drifter_lat = ds_drifter.latitude.values

    # Shore distances (longitude, latitude and distance to the shore)
    raster_lon = df_raster.longitude.values
    raster_lat = df_raster.latitude.values
    raster_dist = df_raster.distance.values

    # Interpolate the drifter locations onto the raster
    start = time.time()
    print('Started interpolation...', end='')
    drifter_dist = griddata((raster_lon, raster_lat), raster_dist, (drifter_lon, drifter_lat), method=method)
    print(f'Done. Elapsed time {np.round(time.time() - start, 2)}s')

    return drifter_dist


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
    if not ds.location_type.values[0]:
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


def get_drogue_presence(ds):
    obs = ds.obs.values
    drogue_lost_dates = ds.drogue_lost_date.values

    ids = ds.ids.values
    IDs = ds.ID.values
    drogue_presence = np.ones(len(obs), dtype=bool)
    for (drogue_lost_date, ID) in zip(drogue_lost_dates, IDs):
        # if drogue lost date is not known, assume it is always present
        if not np.isnat(drogue_lost_date):
            obs_id = obs[ids == ID]
            times = ds.time.values[obs_id]
            drogue_presence[obs_id] = np.where(times > drogue_lost_date, False, True)

    return drogue_presence
