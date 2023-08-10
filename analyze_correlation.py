import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="iteritems")
import toolbox as tb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from file_names import *

y_column = 'beaching_flag'

df = pd.read_csv(file_name_4, parse_dates=['time_start', 'time_end'], index_col='ID')
df.drop(columns=['ID.2', 'ID.1', 'longitude_start', 'latitude_start', 'longitude_end', 'latitude_end',
                 'time_start', 'time_end', 'inproduct_wind_nearest_shore'], inplace=True)
df['speed'] = np.hypot(df['velocity_north'], df['velocity_east'])
df.rename(columns={'velocity_north': 'velocity_v', 'velocity_east': 'velocity_u',
                   'shortest_distance_n': 'shore_distance_y', 'shortest_distance_e': 'shore_distance_x',
                   }, inplace=True)
# Renaming columns based on the mentioned conditions
new_columns = {col: col.replace('score_', 'scr_').replace('deg', '').replace('km', '')
               for col in df.columns if col.startswith('score')}

# Rename the DataFrame columns
df.rename(columns=new_columns, inplace=True)

df['inprod_u10m_d'] = tb.get_unit_inproducts(df['shore_distance_x'].values,
                                                       df['shore_distance_y'].values,
                                                         df['wind10m_u_mean'].values,
                                                         df['wind10m_v_mean'].values,
                                                       unit=False)
df['inprod_unit_u10m_d'] = tb.get_unit_inproducts(df['shore_distance_x'].values,
                                                            df['shore_distance_y'].values,
                                                            df['wind10m_u_mean'].values,
                                                            df['wind10m_v_mean'].values)
df['inprod_v_d'] = tb.get_unit_inproducts(df['shore_distance_x'].values, df['shore_distance_y'].values,
                                                        df['velocity_u'].values, df['velocity_v'].values,
                                                        unit=False)
df['inprod_unit_v_d'] = tb.get_unit_inproducts(df['shore_distance_x'].values, df['shore_distance_y'].values,
                                                        df['velocity_u'].values, df['velocity_v'].values)


#%% filter out drifters with more than 15 false beaching flags and more than 6 true beaching flags
table_beaching_per_drifter = df.groupby('drifter_id').beaching_flag.value_counts().unstack().fillna(0).astype(int)
drifter_ids_to_keep = table_beaching_per_drifter[
    (table_beaching_per_drifter[False] <= 15) & (table_beaching_per_drifter[True] <= 6)].index
df = df[df.drifter_id.isin(drifter_ids_to_keep)]

df.drop(columns=['drifter_id'], inplace=True)

#%%
cor = df.corr(numeric_only=True)
diki = cor[y_column].sort_values(ascending=True)[:-1]

import numpy as np


def compute_confidence_interval(r, n, alpha=0.05):
    z = 0.5 * np.log((1 + r) / (1 - r))
    SE = 1 / np.sqrt(n - 3)

    # Compute z values for lower and upper tails
    z_crit = np.abs(np.percentile(np.random.standard_normal(10000), [alpha / 2 * 100, 100 - alpha / 2 * 100]))
    lower_z, upper_z = z - z_crit[1] * SE, z + z_crit[0] * SE

    # Convert back to r
    lower_r = (np.exp(2 * lower_z) - 1) / (np.exp(2 * lower_z) + 1)
    upper_r = (np.exp(2 * upper_z) - 1) / (np.exp(2 * upper_z) + 1)

    return lower_r, upper_r

plt.figure(figsize=(7, 12))

# Example
n = len(df)
lower_ci, upper_ci = zip(*diki.apply(lambda r: compute_confidence_interval(r, n)))

plt.figure(figsize=(7, 12))
plt.barh(diki.index, diki, xerr=(diki - lower_ci, upper_ci - diki),
         capsize=5)

plt.xlabel('Correlation with grounding flag')
plt.grid(axis='x')
#plot horizontal thin lines from yticks to origin
x_lims = plt.gca().get_xlim()
for y in diki.index:
    plt.plot([x_lims[0], 0], [y, y], 'k--', alpha=0.3)

plt.xlim(x_lims)

plt.tight_layout()

plt.savefig(f'figures/corr_coef_grounding_filtered.png', dpi=300)

plt.show()

#%% make histogram for all features grouped by grounding flag

# sort columns in df alphabetically
df = df.reindex(sorted(df.columns), axis=1)

unique_flags = df['beaching_flag'].unique()

for flag in unique_flags:
    plt.figure(figsize=(15, 15))

    subset_df = df[df['beaching_flag'] == flag].drop(columns=new_columns.values())
    subset_df.hist(figsize=(15, 15), bins=50)

    # Adjust title size of all subplots
    for ax in plt.gcf().axes:
        ax.set_title(ax.get_title(), fontsize=10)

    plt.suptitle(f'Histograms for {("positive" if flag else "negative")} grounding flag', fontsize=16)  # Add a centered title for clarity
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate the suptitle

    plt.savefig(f'figures/histograms_grounding_flag_{flag}.png', dpi=300)

    plt.show()

