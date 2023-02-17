import load_data
import numpy as np
import analyzer
import pickle_manager as pickm
from plotter import *


def load_random_subset():
    ds = load_data.get_ds_drifters()

    n = len(ds.traj)
    traj = np.random.randint(0, len(ds.traj), int(n/100))
    obs = np.where(np.isin(ds.ids, ds.ID[traj]))[0]

    return ds.isel(traj=traj, obs=obs)


ds_subset = pickm.pickle_wrapper('random_subset_1', load_random_subset)

analyzer.count_death_codes(ds_subset, verbose=True)
