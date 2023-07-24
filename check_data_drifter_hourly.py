import matplotlib.pyplot as plt
import numpy as np
import toolbox as tb
import plotter
import load_data


ds = load_data.get_ds_drifters()

print(ds.info())

for item in ds.data_vars.items():
    print(f'{item[0]} :\t {ds[item[0]].attrs}\n')
    # if ds[item[0]].dims[0] == 'traj':
    #     try:
    #         plot_uniques_bar(ds[item[0]], xlabel=ds[item[0]].attrs['long_name'])
    #     except:
    #         pass

#%%
plotter.plot_death_type_bar(ds)

times = tb.days_without_drogue(ds)
plt.figure(figsize=(10, 6))
plt.hist(times, bins=100)
plt.xlabel('time lost drogue [days]')

plt.ylabel('# drifters')

# plt.yscale('log')

plt.savefig('figures/time_lost_drogue.png', dpi=300)

plt.show()


#%%
def find_last_points(ds):
    index_last_points = []
    id_prev = ds.ID.values[0]
    for i, ID in enumerate(ds.ids.values):
        if ID != id_prev:
            index_last_points.append(i-1)
        id_prev = ID
    return index_last_points


last_points = find_last_points(ds)
plotter.plot_trajectories_death_type(ds.isel(obs=last_points), s=5)
