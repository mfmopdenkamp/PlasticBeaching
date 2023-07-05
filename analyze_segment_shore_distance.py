from file_names import file_name_4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import toolbox as tb

df = pd.read_csv(file_name_4, parse_dates=['time_start', 'time_end'], index_col='ID')
delta_score = 0.02
shore_score_thresholds = np.arange(0, 1+delta_score, delta_score)

#%% filter out drifters with less more than 15 false beaching flags and more than 6 true beaching flags
table_beaching_per_drifter = df.groupby('drifter_id').beaching_flag.value_counts().unstack().fillna(0).astype(int)
drifter_ids_to_keep = table_beaching_per_drifter[(table_beaching_per_drifter[False] <= 15) & (table_beaching_per_drifter[True] <= 6)].index

df = df[df.drifter_id.isin(drifter_ids_to_keep)]

#%% plot beaching probability vs shore distance
shore_distance_thresholds = np.arange(0, 15000, 500)
beaching_prob_by_shore_distance = np.zeros(len(shore_distance_thresholds))

for i, shore_distance_threshold in enumerate(shore_distance_thresholds):
    df_filtered = df[df.shortest_distance <= shore_distance_threshold]
    beaching_prob_by_shore_distance[i] = df_filtered.beaching_flag.mean()

plt.figure(figsize=(7, 5))
plt.plot(shore_distance_thresholds/1000, beaching_prob_by_shore_distance, 'o-')
plt.xlabel('shore distance threshold [km]')
plt.ylabel('Beaching probability')
plt.grid()
plt.savefig('figures/beaching_probability_vs_shore_distance_threshold.png', dpi=300)
plt.show()

#%% define plot functions
colors = ['tab:black', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

def plot_360(c_names, beaching_probs, title='', style='-', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    for i, col in enumerate(c_names):
        if col == 'all':
            ax.plot(shore_score_thresholds, beaching_probs[col], 'ko'+style, label=col.split('_')[-1])
        else:
            ax.plot(shore_score_thresholds, beaching_probs[col], 'o'+style, label=col.split('_')[-1])

    ax.set_title(title)
    ax.grid()


def plot_wind(c_names, beaching_probs, title='', style='o-', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    for i, col in enumerate(c_names):
        if col == 'all':
            ax.plot(shore_score_thresholds, beaching_probs[col], style, label=col.split('_')[-1])
        else:
            ax.plot(shore_score_thresholds, beaching_probs[col], style, color=colors[i],
                    label=col.split('_')[-2]+'_'+col.split('_')[-1])

    ax.set_title(title)
    ax.grid()


#%%
fig, axs = plt.subplots(4, 2, figsize=(10, 15), sharex=True, sharey=True)


types = ['shore', 'beach', 'bedrock', 'wetland']
for j, type in enumerate(types):
    column_names = [f'score_{type}_360deg_10km', f'score_{type}_360deg_8km', f'score_{type}_360deg_5km', f'score_{type}_360deg_2km']
    ground_probs_smaller, ground_probs_larger = tb.get_probabilities(df, column_names=column_names, score_thresholds=shore_score_thresholds)
    i = 0
    plot_360(column_names, ground_probs_smaller, title=('within full circle' if j == 0 else ''), ax=axs[j, i])
    # ax = axs[j, i].twinx()
    # plot_360(column_names, ground_probs_larger, style='^:', ax=ax)

    if j == 0:
        axs[j, i].legend(title='radius', loc='upper left')
        # ax.legend(title='radius', loc='bottom right')
    elif j == 3:
        axs[j, i].set_xlabel('normalized score threshold')
    axs[j, i].set_ylabel(fr'$\mathbf{{{type}}}$')
                         # + '\nbeaching probability')

    column_names = [f'score_{type}_22deg_14km', f'score_{type}_45deg_10km', f'score_{type}_60deg_9km', f'score_{type}_180deg_5km',
                    f'score_{type}_270deg_4km']
    ground_probs_smaller, ground_probs_larger = tb.get_probabilities(df, column_names=column_names, score_thresholds=shore_score_thresholds)
    i = 1
    plot_wind(column_names, ground_probs_smaller, title=('within circle slice' if j == 0 else ''), ax=axs[j, i])
    # ax = axs[j, i].twinx()
    # plot_wind(column_names, ground_probs_larger, style='^:', ax=ax)

    if i == 0:
        axs[j, i].set_ylabel('beaching probability')
    if j == 0:
        axs[j, i].legend(title='slice shape', loc='upper left')
        # ax.legend(title='radius', loc='bottom right')
    elif j == 3:
        axs[j, i].set_xlabel('normalized score threshold')

# Set the common y-label for the entire figure
fig.text(0, 0.5, 'beaching probability', va='center', rotation='vertical')
plt.tight_layout()
plt.savefig('figures/grounding_prob_vs_score_threshold.png', dpi=300)
plt.show()
