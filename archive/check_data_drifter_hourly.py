import matplotlib.pyplot as plt
import numpy as np

import analyzer
import plotter
import load_data


ds = load_data.get_ds_drifters('gdp_random_subset_1')

for item in ds.data_vars.items():
    print(f'{item[0]} :\t {ds[item[0]].attrs}\n')
    # if ds[item[0]].dims[0] == 'traj':
    #     try:
    #         plot_uniques_bar(ds[item[0]], xlabel=ds[item[0]].attrs['long_name'])
    #     except:
    #         pass

plotter.plot_death_type_bar(ds)

times = analyzer.days_without_drogue(ds)

plt.hist(times[np.where((times < 1000) * (times > 0))], bins=100)
plt.xlabel('time lost drogue [days]')
# plt.xlim([0, 100])
plt.ylabel('# drifters')
# plt.semilogy()
plt.show()


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
