import numpy as np
import geopandas as gpd
from tqdm import tqdm
import time
from scipy.interpolate import griddata


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

    # create geopandas dataframe from xarray dataset
    gdf_gdp = gpd.GeoDataFrame({'latitude': ds_gdp.latitude, 'longitude': ds_gdp.longitude},
                              geometry=gpd.points_from_xy(ds_gdp.longitude, ds_gdp.latitude),
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


def determine_beaching_event(distance, velocity, max_distance_m, max_velocity_mps):
    if len(distance) != len(velocity):
        raise ValueError('distance and velocity array must have the same length!')

    beaching_rows = np.zeros(len(distance), dtype=bool)

    count = 0
    threshold_h = 4
    for i, (d, v) in enumerate(zip(distance, velocity)):
        if d < max_distance_m and v < max_velocity_mps:
            count += 1

            if count >= threshold_h:
                beaching_rows[i] = True

        else:
            count = 0

    return beaching_rows


def end_date_2_years(ds_end_date):
    return ds_end_date.values.astype('datetime64[Y]').astype(int) + 1970


def find_index_last_coord(ds):
    index_last_coord = []
    ids = ds.ids.values

    id_prev = ids[0]
    for i, ID in enumerate(ids):
        if ID != id_prev:
            index_last_coord.append(i-1)
        id_prev = ID
    index_last_coord.append(ids[-1])

    return index_last_coord


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


def drogue_presence(ds):

    drogue_presence = np.ones(len(ds.obs), dtype=bool)
    drogue_lost_dates = ds.drogue_lost_date.values

    ids = ds.ids.values
    IDs = ds.ID.values
    for i_traj, (drogue_lost_date, ID) in enumerate(zip(drogue_lost_dates, IDs)):
        # if drogue lost date is not known, assume it is always present
        if not np.isnat(drogue_lost_date):
            obs = np.where(ids == ID)[0]
            times = ds.time.values[obs]
            drogue_presence[obs] = np.where(times > drogue_lost_date, False, True)

    return drogue_presence


if __name__ == '__main__':
    import load_data
    ds = load_data.get_ds_drifters('gdp_random_subset_1')

    drogue_presence = drogue_presence(ds)