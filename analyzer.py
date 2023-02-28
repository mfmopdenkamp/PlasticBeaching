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

