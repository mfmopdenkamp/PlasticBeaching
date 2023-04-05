import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
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


def get_mask_drifter_on_shore(distance, velocity, max_distance_m, max_velocity_mps, threshold_count=4):
    if len(distance) != len(velocity):
        raise ValueError('distance and velocity array must have the same length!')

    beaching_tags = np.zeros(len(distance), dtype=bool)

    count = 0
    threshold_count = threshold_count
    for i, (d, v) in enumerate(zip(distance, velocity)):
        if d < max_distance_m and v < max_velocity_mps:
            count += 1

            if count == threshold_count:
                beaching_tags[i-threshold_count:i] = True
            elif count > threshold_count:
                beaching_tags[i] = True

        else:
            count = 0

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

            beaching_rows = get_mask_drifter_on_shore(ds_i.aprox_distance_shoreline.values[-10:],
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


def drogue_presence(ds):

    drogue_presence = np.ones(len(ds.obs), dtype=bool)
    drogue_lost_dates = ds.drogue_lost_date.values

    ids = ds.ids.values
    IDs = ds.ID.values
    for (drogue_lost_date, ID) in zip(drogue_lost_dates, IDs):
        # if drogue lost date is not known, assume it is always present
        if not np.isnat(drogue_lost_date):
            obs = np.where(ids == ID)[0]
            times = ds.time.values[obs]
            drogue_presence[obs] = np.where(times > drogue_lost_date, False, True)

    return drogue_presence


if __name__ == '__main__':
    import load_data
    import plotter
    ds = load_data.get_ds_drifters('gdp_random_subset_1')

    last_coords = find_index_last_obs_traj(ds)

    drogue_presence = drogue_presence(ds)
    obs = np.where(ds.latitude > 0)[0]
    traj = traj_from_obs(ds, obs)
    plotter.plot_death_type_bar(ds)
    plotter.plot_trajectories_death_type(ds.isel(traj=traj, obs=obs))
