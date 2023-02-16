import pickle
import load_data
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import plotter
import time
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


if __name__ == '__main__':
    proximity = 10
    ds = load_data.get_ds_drifters(proximity_of_coast=proximity, with_distances=False)
    traj = np.where(ds.type_death == 1)[0]
    obs = np.where(np.isin(ds.ids, ds.ID[traj]))[0]
    ds = ds.isel(traj=traj, obs=obs)

    resolution = 'f'
    df_shore = load_data.get_shoreline(resolution)

    ds['distance_shoreline'] = ('obs', find_shortest_distance(ds, df_shore))

    with open(f'pickledumps/ds_gdp_subset_{proximity}km_res_{resolution}.pkl', 'wb') as f:
        pickle.dump(ds, f)


    # PLOTTING
    def plot():
        fig, ax = plotter.get_sophie_subplots(extent=(-92, -88.5, -1.75, 1), title=f'Trajectory drifter')
        pcm = ax.scatter(ds.longitude, ds.latitude, transform=ccrs.PlateCarree(),
                         c=ds.distance_shoreline, cmap='inferno')
        ax.scatter(ds.longitude, ds.latitude, transform=ccrs.PlateCarree(), c='k', s=0.4,
                   label='all', alpha=0.8)
        df_shore.plot(ax=ax)
        cb = fig.colorbar(pcm, ax=ax)
        cb.set_label('Distance to shore')
        ax.set_ylabel('Latitude N')
        ax.set_xlabel('Longitude E')
        plt.show()


    plot()


    # ANALYSIS
    ds_shortest = ds_gdp.isel(row=np.argmin(ds_gdp.distance_shoreline.values))

    print(ds_shortest.dims)
