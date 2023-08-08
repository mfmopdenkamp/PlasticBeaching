from numba import jit
import picklemanager as pickm
import numpy as np
import matplotlib.pyplot as plt
import toolbox as tb

ds = pickm.load_pickle(pickm.create_pickle_path('gdp_drogue_presence'))
ds_undrogued = ds.isel(obs=ds.obs[~ds.drogue_presence.values])
ds_drogued = ds.isel(obs=ds.obs[ds.drogue_presence.values])

#%%
n_obs = len(ds.obs)
n_traj = len(ds.traj)


@jit(nopython=True)
def get_count(ds_ap_dist, ds_ids, distances):
    trajs = np.zeros(len(distances))
    obss = np.zeros(len(distances))

    for i, distance in enumerate(distances):
        mask = ds_ap_dist < distance

        trajs[i] = len(np.unique(ds_ids[mask]))
        obss[i] = np.sum(mask)

    return trajs, obss


distances_far = np.flip(np.arange(5, 1000, 50))
trajs_far_drogued, obss_far_drogued = get_count(ds_drogued.aprox_distance_shoreline.values, ds_drogued.ids.values,
                                                distances_far)
trajs_far_undrogued, obss_far_undrogued = get_count(ds_undrogued.aprox_distance_shoreline.values,
                                                    ds_undrogued.ids.values, distances_far)

distances_close = np.flip(np.arange(1, 16, 0.5))
trajs_close_drogued, obss_close_drogued = get_count(ds_drogued.aprox_distance_shoreline.values, ds_drogued.ids.values,
                                                distances_close)
trajs_close_undrogued, obss_close_undrogued = get_count(ds_undrogued.aprox_distance_shoreline.values,
                                                    ds_undrogued.ids.values, distances_close)
#%%
pickm.dump_pickle([trajs_far_drogued, obss_far_drogued, trajs_far_undrogued, obss_far_undrogued,
                     trajs_close_drogued, obss_close_drogued, trajs_close_undrogued, obss_close_undrogued],
                  base_name='results_drogue_vs_undrogue_traj_obs_count')


#%%
fig, axs = plt.subplots(2, 2, sharex='col', figsize=(10, 10), dpi=300)


def plot_ax(ax, dists, drogue_portion, undrogue_portion, show_legend=False, ylabel=''):
    ax.plot(dists, drogue_portion, 'o-', label='Drogued')
    ax.plot(dists, undrogue_portion, 's-', label='Undrogued')
    ax.set_ylabel(ylabel)

    if show_legend:
        ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.plot(dists, undrogue_portion / (drogue_portion + undrogue_portion), 'b^-',
             label='Fraction undrogued')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Fraction undrogued')
    if show_legend:
        ax2.legend(loc='upper right')


plot_ax(axs[0, 0], distances_far, trajs_far_drogued, trajs_far_undrogued, show_legend=True, ylabel='Number of trajectories')
plot_ax(axs[0, 1], distances_close, trajs_close_drogued, trajs_close_undrogued, ylabel='Number of trajectories')
plot_ax(axs[1, 0], distances_far, obss_far_drogued, obss_far_undrogued, ylabel='Number of observations')
plot_ax(axs[1, 1], distances_close, obss_close_drogued, obss_close_undrogued, ylabel='Number of observations')


fig.text(0.5, 0.07, 'Maximum distance to the shoreline (km)', ha='center')

axs[0, 0].set_title(r'\textbf{Far} from the shore')
axs[0, 1].set_title('Close to the shore')

for ax in axs.flat:
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.grid(axis='x')

plt.subplots_adjust(hspace=0.1, wspace=0.3)

plt.savefig('figures/drogue_vs_undrogue_distance_fractions.png')

plt.show()