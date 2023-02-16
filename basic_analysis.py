import load_data
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

ds = load_data.get_ds_drifters()

#%%
# select only death type == 1
traj = np.where(ds.type_death == 1)[0]
obs = np.where(np.isin(ds.ids, ds.ID[traj]))[0]
ds = ds.isel(traj=np.where(ds.type_death == 1)[0], obs=obs)


ids = np.unique(ds.ids)
n = len(ids)


def plot_last_distances():
    last_hours = 100
    max_drifters = 9
    fig, ax = plt.subplots()
    for i, ID in enumerate(tqdm(ids)):
        ds_id = ds.isel(obs=np.where(ds.ids == ID)[0])
        distance = ds_id.distance_shoreline.values[:-last_hours:-1]
        plt.plot(np.arange(len(distance)), distance/1000, label=str(ID))
        if i == max_drifters:
            break

    ax.set_xlabel('hours to grounding event')
    ax.set_ylabel('distance to the shoreline [km]')
    ax.set_ylim([0, last_hours])

    plt.show()


plot_last_distances()


fig, ax = plt.subplots()
ax.hist(np.hypot(ds.ve, ds.vn), bins=50, density=True)
ax.set_xlabel('velocity [m/s]')
plt.show()
