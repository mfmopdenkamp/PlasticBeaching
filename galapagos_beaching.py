import picklemanager as pickm
from plotter import *
from toolbox import get_obs_drifter_on_shore

pickle_name = pickm.create_pickle_path('ds_galapagos_distance')
ds_g = pickm.load_pickle(pickle_name)

mask_drifter_beached = get_obs_drifter_on_shore(ds_g.distance_shoreline.values, np.hypot(ds_g))

ds_g_s = ds_g.isel(obs=mask_drifter_beached)
plot_galapagos_map_distances(ds_g_s, title='Galapagos, beached instances')

# traj = np.where(np.isin(ds_g.ID, np.unique(ds_g.ids.isel(obs=trapping_rows))))[0]
# ds_g_trap = ds_g.isel(obs=trapping_rows, traj=traj)
#
# plot_death_type_bar(ds_g_trap)