from file_names import file_name_4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import toolbox as tb

df = pd.read_csv(file_name_4, parse_dates=['time_start', 'time_end'], index_col='ID')
delta_score = 0.02
tidal_elevation_split_points = np.arange(0, 1 + delta_score, delta_score)

#%% filter out drifters with less more than 15 false beaching flags and more than 6 true beaching flags
table_beaching_per_drifter = df.groupby('drifter_id').beaching_flag.value_counts().unstack().fillna(0).astype(int)
drifter_ids_to_keep = table_beaching_per_drifter[(table_beaching_per_drifter[False] <= 15) & (table_beaching_per_drifter[True] <= 6)].index

df = df[df.drifter_id.isin(drifter_ids_to_keep)]

#%% define plot functions


def plot_360(c_names, beaching_probs, title='', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    for i, col in enumerate(c_names):
        if col == 'all':
            ax.plot(tidal_elevation_split_points, beaching_probs[col], tb.markers[i]+'-',
                    color=tb.colors[i], label='all')
        else:
            ax.plot(tidal_elevation_split_points, beaching_probs[col], tb.markers[i]+'-',
                    color=tb.colors[i], label=col.split('_')[0])

    ax.set_title(title)
    ax.grid()


#%%
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

column_names = ['m2_tidal_elevation_mm', 'm4_tidal_elevation_mm',
   's2_tidal_elevation_mm', 'n2_tidal_elevation_mm',
   'k1_tidal_elevation_mm', 'k2_tidal_elevation_mm',
   'o1_tidal_elevation_mm', 'p1_tidal_elevation_mm',
   'q1_tidal_elevation_mm']

ground_probs_smaller, ground_probs_larger = tb.get_probabilities(df, column_names=column_names,
                                                                 split_points=tidal_elevation_split_points)
plot_360(column_names, ground_probs_smaller, ax=ax)
ax.legend(title='tidal constituent')

fig.text(0, 0.5, 'beaching probability smaller than threshold', va='center', rotation='vertical')
plt.xlabel('split point normalized tidal elevation')
plt.tight_layout()
plt.savefig('figures/grounding_prob_vs_tidals_smaller.png', dpi=300)
plt.show()
#%%
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
plot_360(column_names, ground_probs_larger, ax=ax)
ax.legend(title='tidal constituent')

fig.text(0, 0.5, 'beaching probability larger than threshold', va='center', rotation='vertical')
plt.xlabel('split point normalized tidal elevation')
plt.tight_layout()
plt.savefig('figures/grounding_prob_vs_tidals_larger.png', dpi=300)
plt.show()

#%% plot impurity reduction
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

column_names = ['m2_tidal_elevation_mm', 'm4_tidal_elevation_mm',
   's2_tidal_elevation_mm', 'n2_tidal_elevation_mm',
   'k1_tidal_elevation_mm', 'k2_tidal_elevation_mm',
   'o1_tidal_elevation_mm', 'p1_tidal_elevation_mm',
   'q1_tidal_elevation_mm']

impurity_reductions = tb.get_impurity_reduction(df, column_names=column_names,
                                                                 split_points=tidal_elevation_split_points)
plot_360(column_names, impurity_reductions, ax=ax)
ax.legend()
# Set the common y-label for the entire figure
plt.ylabel('gini impurity reduction')
plt.xlabel('split point normalized tidal elevation')
plt.tight_layout()
plt.savefig('figures/impurity_reduction_tidals', dpi=300)
plt.show()
