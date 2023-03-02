import load_data
import numpy as np
import matplotlib.pyplot as plt
import picklemanager as pickm
from plotter import *
from analyzer import *
import xarray as xr
import time
from tqdm import tqdm


def load_random_subset():
    ds = load_data.get_ds_drifters()

    n = len(ds.traj)
    traj = np.random.choice(np.arange(len(ds.traj)), size=int(n/100), replace=False)
    obs = obs_from_traj(ds, traj)
    ds_subset = ds.isel(traj=traj, obs=obs)
    df_raster = load_data.get_raster_distance_to_shore_04deg()
    ds_subset['aprox_distance_shoreline'] = xr.DataArray(
        data=interpolate_drifter_location(df_raster, ds_subset, method='nearest'),
        dims='obs',
        attrs={'long_name': 'Approximate distance to shoreline by interpolation onto 0.04deg raster',
               'units': 'km'})
    return ds_subset


ds = pickm.pickle_wrapper('gdp_random_subset_5', load_random_subset)

plot_death_type_bar(ds)

close_2_shore = ds.aprox_distance_shoreline < 10
obs = np.where(close_2_shore)[0]

plot_trajectories_death_type(ds.isel(obs=obs, traj=traj_from_obs(ds, obs)))

test1 = np.array([0,1,1,1,0,0,1,1,1,0,0,1,1,1,1,1,0], dtype=bool)
test2 = np.array([1,0,1,0,1,1,0,0,1], dtype=bool)

def get_event_indexes(mask):
    mask = mask.astype(int)
    i_start = np.where(np.diff(mask) == 1)[0] + 1
    i_end = np.where(np.diff(mask) == -1)[0] + 1

    if mask[0] and not mask[1]:
        i_start = np.insert(i_start, 0, 0)
    if mask[-1]:
        i_end = np.append(i_end, len(mask))
    return i_start, i_end

i_s, i_e = get_event_indexes(close_2_shore)

