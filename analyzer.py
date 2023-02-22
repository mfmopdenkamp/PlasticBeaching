import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from tqdm import tqdm


def find_shortest_distance(ds_gdp, gdf_shoreline):
    """"Calculates distance from shoreline (Polygons) to drifters for every drifters location (Point)"""

    # create geopandas dataframe from xarray dataset
    gdf_gdp = gpd.GeoDataFrame({'latitude': ds_gdp.latitude, 'longitude': ds_gdp.longitude},
                              geometry=gpd.points_from_xy(ds_gdp.longitude, ds_gdp.latitude),
                              crs='epsg:4326')

    gdf_shoreline.to_crs(crs=3857, inplace=True)
    gdf_gdp.to_crs(gdf_shoreline.crs, inplace=True)

    dtype = np.float32
    init_distance = np.finfo(np.float32).max
    shortest_distances = np.ones(len(gdf_gdp), dtype=dtype) * init_distance

    for i, point in enumerate(tqdm(gdf_gdp.geometry)):
        for polygon in gdf_shoreline.geometry:
            distance = point.distance(polygon)
            if distance < shortest_distances[i]:
                shortest_distances[i] = distance

    return shortest_distances


def determine_beaching_event_distance(ds):
    trapping_rows = np.empty(0, dtype=int)
    for ID in ds.ID:
        rows = np.where(ds.ids == ID)[0]
        distance = ds.distance_shoreline[rows]
        velocity = np.hypot(ds.vn[rows], ds.ve[rows])

        count = 0
        threshold_h = 4
        for i, (d, v) in enumerate(zip(distance, velocity)):
            if d < 500 and v < 0.1:
                count += 1
            else:
                if count >= threshold_h:
                    trapping_rows = np.append(trapping_rows, rows[i - count:i])
                count = 0

    return trapping_rows


if __name__ == '__main__':
    import load_data
    ds = load_data.get_ds_drifters(proximity_of_coast=10)

    count_death_codes(ds, verbose=True)