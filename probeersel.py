import numpy as np
import load_data

ds = load_data.load_subset(traj_percentage=2)

ds_argos = load_data.load_subset(location_type='argos', ds=ds)

print(np.unique(ds.location_type.values))
print(np.unique(ds_argos.location_type.values))

ds_undrogued = load_data.load_subset(drogued=True, ds=ds)
ds_drogued = load_data.load_subset(drogued=False, ds=ds)

print(len(ds.obs))
print(len(ds_undrogued.obs))
print(len(ds_drogued.obs))
print(len(ds.obs) == len(ds_undrogued.obs) + len(ds_drogued.obs))
