import pandas as pd
import load_data
import pickle_manager as pickm
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import time


def interpolate_drifter_location(df_shore, ds_drifter):
    # Drifter locations (longitude and latitude)
    drifter_lon = ds_drifter.longitude.values
    drifter_lat = ds_drifter.latitude.values

    # Shore distances (longitude, latitude and distance to the shore)
    shore_lon = df_shore['longitude'].values
    shore_lat = df_shore['latitude'].values
    shore_dist = df_shore['distance'].values

    # Interpolate the drifter locations onto the raster
    start = time.time()
    drifter_dist = griddata((shore_lon, shore_lat), shore_dist, (drifter_lon, drifter_lat), method='nearest')
    print(f'Interpolation done. Elapsed time {np.round(time.time() - start, 2)}s')

    return drifter_dist


if __name__ == '__main__':
    filename = 'dist2coast.txt.bz2'
    df_shore = pickm.load_pickle_wrapper(filename, pd.read_csv, load_data.data_folder+filename,
                                       delim_whitespace=True, names=['longitude', 'latitude', 'distance'],
                                       header=None, compression='bz2')

    filename = 'gdp_v2.00.nc'
    ds = pickm.load_pickle_wrapper(filename, load_data.drifter_data_hourly, filename)


    drifter_dist = interpolate_drifter_location(df_shore, ds)

    proximity = 10  # kilometers
    close_to_shore = drifter_dist < proximity
    print(close_to_shore.sum())

    # Plot distance to shore distribution. Why are they so close to the shore?
    fig1 = plt.figure()
    plt.hist(df_shore['distance'].values, bins=50)
    plt.xlabel('distance to shore [km]')
    plt.ylabel('number raster points')
    plt.show()

    raster_dist_deg = 0.04
    circ_earth = 40075
    max_dist_deg = np.sqrt(2 * raster_dist_deg ** 2)
    max_dist_km = circ_earth * max_dist_deg / 360




