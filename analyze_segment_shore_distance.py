from file_names import file_name_4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import toolbox as tb

df = pd.read_csv(file_name_4, parse_dates=['time_start', 'time_end'], index_col='ID')
shore_score_thresholds = np.arange(0, 1.1, 0.02)

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


def plot_360(c_names, beaching_probs, title='', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    for col in c_names:
        if col == 'all':
            plt.plot(shore_score_thresholds, beaching_probs[col], 'ko-', label=col.split('_')[-1])
        else:
            plt.plot(shore_score_thresholds, beaching_probs[col], 'o-', label=col.split('_')[-1])

    ax.legend(title='radius')
    ax.set_xlabel('normalized score threshold')
    ax.set_ylabel('beaching probability')
    ax.set_title(title)
    ax.grid()


def plot_wind(c_names, beaching_probs, title='', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    for col in c_names:
        if col == 'all':
            plt.plot(shore_score_thresholds, beaching_probs[col], 'ko-', label=col.split('_')[-1])
        elif col == 'score_shore_270deg_4km':
            plt.plot(shore_score_thresholds, beaching_probs[col], 'o-', c='cyan', label=col.split('_')[-2]+'_'+col.split('_')[-1])
        else:
            plt.plot(shore_score_thresholds, beaching_probs[col], 'o-', label=col.split('_')[-2]+'_'+col.split('_')[-1])

    ax.legend(title='radius')
    ax.set_xlabel('normalized score threshold')
    ax.set_ylabel('beaching probability')
    ax.set_title(title)
    ax.grid()


#%%

fig, axs = plt.figure(figsize=(20, 10), sharex='col')

i = 1
types = ['shore', 'beach', 'bedrock', 'wetland']
for type in types:
    column_names = [f'score_{type}_360deg_10km', f'score_{type}_360deg_8km', f'score_{type}_360deg_5km', f'score_{type}_360deg_2km']
    beaching_probs_shore = tb.get_probabilities(df, column_names=column_names)

    ax = axs[i // 2, i % 2]
    plot_360(column_names, beaching_probs_shore, title='360deg '+type, ax=ax)

    i += 1
    column_names = [f'score_{type}_22deg_14km', f'score_{type}_45deg_10km', f'score_{type}_60deg_9km', f'score_{type}_180deg_5km',
                    f'score_{type}_270deg_4km']
    beaching_probs_shore = tb.get_probabilities(df, column_names=column_names)

    ax = axs[i // 2, i % 2]
    plot_wind(column_names, beaching_probs_shore, title='wind '+type, ax=ax)
    i += 1

plt.savefig('figures/grounding_prob_vs_score_threshold.png', dpi=300)
plt.show()