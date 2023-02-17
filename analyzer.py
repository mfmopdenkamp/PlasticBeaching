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


def count_death_codes(ds, verbose=False):
    death_types = np.unique(ds.type_death)
    n_death_types = np.zeros(len(death_types))
    for i_death, death_type in enumerate(death_types):
        n_death_types[i_death] = sum(ds.type_death == death_type)

    if verbose:
        fig, ax = plt.subplots()
        ax.bar(death_types, n_death_types)
        ax.set_xlabel('death type')
        ax.set_ylabel('# drifters')
        plt.xticks(death_types)
        plt.show()

    return death_types, n_death_types


if __name__ == '__main__':
    import load_data
    ds = load_data.get_ds_drifters(proximity_of_coast=10)

    count_death_codes(ds, verbose=True)