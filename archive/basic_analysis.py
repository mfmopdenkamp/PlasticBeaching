from plotter import *
from analyzer import *

ds = load_data.get_ds_drifters()

#%%
# select only death type == 1
traj = np.where(ds.type_death == 1)[0]
obs = obs_from_traj(ds, traj)
ds = ds.isel(traj=traj, obs=obs)

plot_last_distances(ds)

plot_velocity_hist(ds)
