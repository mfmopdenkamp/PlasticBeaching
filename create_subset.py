import pandas as pd
import load_data
import pickle
import pickle_manager as pickm
import numpy as np
import matplotlib.pyplot as plt
from analyzer import interpolate_drifter_location


def create_subset(proximity, filename='gdp_v2.00.nc'):
    """proximity is in km"""
    # Load the 0.04deg raster with distances to the shoreline
    df_raster = load_data.get_raster_distance_to_shore_04deg()

    # Load the hourly data from the Global Drifter Program
    ds = load_data.get_ds_drifters(filename=filename)

    # Interpolate the drifter data onto the raster with distances to the shoreline (or load from pickle. Operation on full dataset cost 812s on my laptop)
    drif_aprox_dist_filename = 'drifter_distances_interpolated_0.04deg_raster'
    drifter_dist_approx = pickm.pickle_wrapper(drif_aprox_dist_filename, interpolate_drifter_location, df_raster, ds)

    # Create a subset of the drifter data that is within a certain proximity of the shoreline
    close_to_shore = drifter_dist_approx < proximity

    obs = np.where(close_to_shore)[0]
    traj = np.where(np.isin(ds.ID, np.unique(ds.ids.isel(obs=obs))))[0]
    ds_subset = ds.isel(traj=traj, obs=obs)
    print(f'Number of rows in original GDP dataset = {ds.obs.shape[0]}. Rows left in subset = {close_to_shore.sum()}. '
          f'This is reduction of {np.round(ds.obs.shape[0] / close_to_shore.sum(), 2)} times the original data.')
    return ds_subset


if __name__ == '__main__':

    proximity = 10  # km
    ds_subset = create_subset(proximity)

    # Write to pickle file for easy use.
    pickle_name = pickm.create_pickle_name(f'gdp_subset_{proximity}km')
    with open(pickle_name, 'wb') as f:
        pickle.dump(ds_subset, f)

    # Plot distance to shore distribution. Why are they so close to the shore?
    df_shore = load_data.get_raster_distance_to_shore_04deg()

    fig1 = plt.figure()
    plt.hist(df_shore['distance'].values, bins=50)
    plt.xlabel('distance to shore [km]')
    plt.ylabel('number raster points')
    plt.show()

    raster_dist_deg = 0.04
    circ_earth = 40075
    max_dist_deg = np.sqrt(2 * raster_dist_deg ** 2)
    max_dist_km = circ_earth * max_dist_deg / 360
