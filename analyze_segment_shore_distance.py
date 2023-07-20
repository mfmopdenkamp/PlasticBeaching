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
delta_distance = 500
shore_distance_thresholds = np.arange(delta_distance, 15000+delta_distance, delta_distance)
beaching_prob_by_shore_distance = np.zeros(len(shore_distance_thresholds))

for i, shore_distance_threshold in enumerate(shore_distance_thresholds):
    df_filtered = df[df.shortest_distance <= shore_distance_threshold]
    beaching_prob_by_shore_distance[i] = df_filtered.beaching_flag.mean()

# Define the exponential decay function
from scipy.optimize import curve_fit


def power_func(x, a, b):
    return a * np.power(x, b)


def exponential_func(x, a, b, c):
    return a * np.exp(-b * x) + c


def hard_coded_exp_fit(x):
    return 0.5 * np.exp(-0.0003 * x) + 0.1


# fit exponential function to probability vs shore distance
# Fit the data to the exponential decay function
a_fit, b_fit = curve_fit(power_func, shore_distance_thresholds, beaching_prob_by_shore_distance,
                         p0=(600, 0.1))[0]
# Generate the fitted curve
y_fit = power_func(shore_distance_thresholds, a_fit, b_fit)

# a_fit2, b_fit2, c_fit2 = curve_fit(exponential_func, shore_distance_thresholds, beaching_prob_by_shore_distance,
#                    p0=(0.5, -0.0003, 0.1))[0]

# y_fit2 = exponential_func(shore_distance_thresholds, a_fit2, b_fit2, c_fit2)

y_fit3 = hard_coded_exp_fit(shore_distance_thresholds)

# Plot the original data and the fitted curve
plt.scatter(shore_distance_thresholds, beaching_prob_by_shore_distance, label='Data')

plt.figure(figsize=(7, 5))
plt.scatter(shore_distance_thresholds/1000, beaching_prob_by_shore_distance, label='Data')
plt.plot(shore_distance_thresholds/1000, y_fit, label='Fitted power function')
# plt.plot(shore_distance_thresholds/1000, y_fit2, label='Fitted exponential function')
plt.plot(shore_distance_thresholds/1000, y_fit3, label='Fitted exponential function')
plt.xlabel('shore distance threshold [km]')
plt.ylabel('Grounding probability')
plt.legend()
plt.grid()
plt.savefig('figures/grounding_probability_vs_shore_distance_threshold.png', dpi=300)
plt.show()
#%%
coast_types = ['shore', 'beach', 'bedrock', 'wetland']

ground_probs = {}
column_names_360 = {}
column_names_slices = {}

for c_type in coast_types:
    column_names_360[c_type] = [f'score_{c_type}_360deg_10km', f'score_{c_type}_360deg_8km', f'score_{c_type}_360deg_5km',
                        f'score_{c_type}_360deg_2km']
    column_names_slices[c_type] = [f'score_{c_type}_22deg_14km', f'score_{c_type}_45deg_10km', f'score_{c_type}_60deg_9km',
                           f'score_{c_type}_180deg_5km', f'score_{c_type}_270deg_4km']
    versions = [0, 1]
    column_names_360_c_type = column_names_360[c_type]
    column_names_slices_c_type = column_names_slices[c_type]

    ground_probs.update({
        f'360_{c_type}_{version}':
            tb.get_probabilities(df, column_names=column_names_360_c_type, split_points=shore_score_thresholds)[version]
        for version in versions
    })

    ground_probs.update({
        f'slices_{c_type}_{version}':
            tb.get_probabilities(df, column_names=column_names_slices_c_type, split_points=shore_score_thresholds)[
                version]
        for version in versions
    })

#%% define plot functions
def plot_360(c_names, beaching_probs, title='', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    for i, col in enumerate(c_names):
        if col == 'all':
            ax.plot(shore_score_thresholds, beaching_probs[col], tb.markers[i]+'-', color=tb.colors[i], label=col.split('_')[-1])
        else:
            ax.plot(shore_score_thresholds, beaching_probs[col], tb.markers[i]+'-', color=tb.colors[i], label=col.split('_')[-1])

    ax.set_title(title)
    ax.grid()


def plot_wind(c_names, beaching_probs, title='', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    for i, col in enumerate(c_names):
        if col == 'all':
            ax.plot(shore_score_thresholds, beaching_probs[col], tb.markers[i]+'-', color=tb.colors[i], label=col.split('_')[-1])
        else:
            ax.plot(shore_score_thresholds, beaching_probs[col], tb.markers[i]+'-', color=tb.colors[i],
                    label=col.split('_')[-2]+'_'+col.split('_')[-1])

    ax.set_title(title)
    ax.grid()


#%%
def plot_probs(version=0):
    fig, axs = plt.subplots(4, 2, figsize=(10, 15), sharex=True, sharey=('row' if version == 0 else True), dpi=300)
    # version 0: plot smaller than splitpoint
    # version 1: plot larger than splitpoint

    for j, c_type in enumerate(coast_types):

        i = 0
        plot_360(column_names_360[c_type], ground_probs[f'360_{c_type}_{version}'],
                 title=('within full circle' if j == 0 else ''), ax=axs[j, i])

        if j == 0:
            axs[j, i].legend(title='radius')
        axs[j, i].set_ylabel(fr'$\mathbf{{{c_type}}}$')

        i = 1
        plot_wind(column_names_slices[c_type], ground_probs[f'slices_{c_type}_{version}'],
                  title=('within circle slice' if j == 0 else ''), ax=axs[j, i])

        if i == 0:
            axs[j, i].set_ylabel('beaching probability')
        if j == 0:
            axs[j, i].legend(title='slice shape')

    # Set the common y-label for the entire figure
    fig.text(0, 0.5, 'beaching probability', va='center', rotation='vertical')
    fig.text(0.4, 0.005, 'normalized score threshold', va='center', rotation='horizontal')
    if version == 1:
        plt.ylim([0,1])
    plt.xlim([0, 1])
    plt.tight_layout(rect=[0, 0.01, 1, 1])
    plt.savefig(f'figures/grounding_prob_vs_score_threshold_{("smaller" if version == 0 else "larger")}.png', dpi=300)
    plt.show()


plot_probs(version=0)
plot_probs(version=1)