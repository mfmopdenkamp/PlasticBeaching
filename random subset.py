import load_data
import numpy as np
import matplotlib.pyplot as plt
import picklemanager as pickm
import pandas as pd
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


def get_event_indexes(mask):
    mask = mask.astype(int)
    i_start = np.where(np.diff(mask) == 1)[0] + 1
    i_end = np.where(np.diff(mask) == -1)[0] + 1

    if mask[0] and not mask[1]:
        i_start = np.insert(i_start, 0, 0)
    if mask[-1]:
        i_end = np.append(i_end, len(mask))
    return i_start, i_end


close_2_shore = ds.aprox_distance_shoreline < 10
event_start_indexes, event_end_indexes = get_event_indexes(close_2_shore)
n = len(event_start_indexes)

def get_beaching_flags(ds, event_start_indexes, event_end_indexes):
    if len(event_start_indexes) != len(event_end_indexes):
        raise ValueError('"event starts" and "event ends" must have equal lengths!')

    beaching_flags = np.zeros(len(event_start_indexes), dtype=bool)
    for i, (i_s, i_e) in enumerate(zip(event_start_indexes, event_end_indexes)):

        distance = np.zeros(i_e - i_s, dtype=int)
        velocity = get_absolute_velocity(ds.isel(obs=slice(i_s, i_e)))
        beaching_rows = determine_beaching_event(distance, velocity, max_distance_m=1, max_velocity_mps=0.01)
        if beaching_rows.sum() > 0:
            beaching_flags[i] = True

    return beaching_flags


beaching_flags = get_beaching_flags(ds, event_start_indexes, event_end_indexes)

df = pd.DataFrame(data={'latitude': ds.latitude[event_start_indexes],
                        'longitude': ds.longitude[event_start_indexes],
                        've': ds.ve[event_start_indexes],
                        'vn': ds.vn[event_start_indexes],
                        'orientation coast east': np.zeros(n),
                        'orientation coast north': np.zeros(n),
                        'beaching_flags': beaching_flags})



