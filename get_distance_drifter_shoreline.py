import pickle_manager as pickm
import load_data
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import plotter
import time


def get_shortest_distance(drifters, shoreline):

    dtype = np.float32
    init_distance = np.finfo(np.float32).max
    shortest_distances = np.ones(len(drifters), dtype=dtype) * init_distance

    for i, point in enumerate(drifters.geometry):
        for polygon in shoreline.geometry:
            distance = point.distance(polygon)
            if distance < shortest_distances[i]:
                shortest_distances[i] = distance

    return shortest_distances


filename = 'gdp_galapagos.nc'
ds_gdp = pickm.load_pickle_wrapper(filename, load_data.drifter_data_hourly, filename)
IDs = np.unique(ds_gdp.ID)[0]
ds_gdp = ds_gdp.where(ds_gdp.ID.isin(IDs), drop=True)

df_gdp = gpd.GeoDataFrame({'latitude' : ds_gdp.latitude, 'longitude': ds_gdp.longitude},
                          geometry=gpd.points_from_xy(ds_gdp.longitude, ds_gdp.latitude),
                          crs='epsg:4326')

resolution = 'c'
df_shore = pickm.load_pickle_wrapper(f'shoreline_{resolution}', load_data.shoreline, resolution)

df_shore.to_crs(crs=3857, inplace=True)
df_gdp.to_crs(df_shore.crs, inplace=True)

print(f'Started to calculate shortest distances for {len(df_gdp)} points..', end='')
start = time.time()
ds_gdp = ds_gdp.assign(distance_shoreline=get_shortest_distance(df_gdp, df_shore))
print(f'Done. Elapsed time {np.round(time.time() - start, 2)}s')


# PLOTTING
def plot():
    fig, ax = plotter.get_sophie_subplots(extent=(-93, -88, -2, 2), title=f'Trajectory drifter {IDs}')
    pcm = ax.scatter(ds_gdp.longitude, ds_gdp.latitude, transform=ccrs.PlateCarree(),
                     c=ds_gdp.distance_shoreline, cmap='inferno')
    ax.scatter(ds_gdp.longitude, ds_gdp.latitude, transform=ccrs.PlateCarree(), c='k', s=0.4,
               label='all', alpha=0.8)
    df_shore.plot(ax=ax)
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label('Distance to shore')
    ax.set_ylabel('Latitude N')
    ax.set_xlabel('Longitude E')
    plt.show()


plot()


# ANALYSIS
ds_shortest = ds_gdp.isel(row = np.argmin(ds_gdp.distance_shoreline.values))

print(ds_shortest.dims)
