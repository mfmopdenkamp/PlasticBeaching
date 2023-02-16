import pickle
import load_data
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import plotter
import time
from shapely.geometry import Point
from tqdm import tqdm


def get_shortest_distance(drifters, shoreline):
    """"Calculates distance from shoreline (Polygons) to drifters for every drifters location (Point)"""
    dtype = np.float32
    init_distance = np.finfo(np.float32).max
    shortest_distances = np.ones(len(drifters), dtype=dtype) * init_distance

    for i, point in enumerate(tqdm(drifters.geometry)):
        for polygon in shoreline.geometry:
            distance = point.distance(polygon)
            if distance < shortest_distances[i]:
                shortest_distances[i] = distance

    return shortest_distances


proximity = 10
ds = load_data.get_ds_drifters(proximity_of_coast=proximity, with_distances=False)
traj = np.where(ds.type_death == 1)[0]
obs = np.where(np.isin(ds.ids, ds.ID[traj]))[0]
ds = ds.isel(traj=traj, obs=obs)

df_gdp = gpd.GeoDataFrame({'latitude': ds.latitude, 'longitude': ds.longitude},
                          geometry=gpd.points_from_xy(ds.longitude, ds.latitude),
                          crs='epsg:4326')

resolution = 'h'
df_shore = load_data.get_shoreline(resolution)

df_shore.to_crs(crs=3857, inplace=True)
df_gdp.to_crs(df_shore.crs, inplace=True)

point1 = df_gdp.geometry[0]
point2 = df_gdp.geometry[1]
distance = point1.distance(point2)