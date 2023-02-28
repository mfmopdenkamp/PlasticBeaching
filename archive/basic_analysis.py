from plotter import *

ds = load_data.get_ds_drifters()

#%%
# select only death type == 1
traj = np.where(ds.type_death == 1)[0]
obs = np.where(np.isin(ds.ids, ds.ID[traj]))[0]
ds = ds.isel(traj=traj, obs=obs)

plot_last_distances(ds)

plot_velocity_hist(ds)
