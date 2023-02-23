import pickle_manager
import pickle_manager as pickm
import numpy as np
from plotter import *
from analyzer import *

pickle_name = pickle_manager.create_pickle_name('ds_galapagos_distance')
ds_g = pickm.load_pickle(pickle_name)

trapping_rows = determine_trapping_event(ds_g)

ds_g_s = ds_g.isel(obs=trapping_rows)
plot_map_distances(ds_g_s, title='Galapagos, beached instances')

traj = np.where(np.isin(ds_g.ID, np.unique(ds_g.ids.isel(obs=trapping_rows))))[0]
ds_g_trap = ds_g.isel(obs=trapping_rows, traj=traj)

plot_death_type_bar(ds_g_trap)
