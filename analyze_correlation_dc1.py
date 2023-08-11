import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="iteritems")
import toolbox as tb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from file_names import *

y_column = 'beaching_flag'

df = pd.read_csv(file_name_4, index_col=None)
df.drop(columns=['longitude', 'latitude', 'time', 'Unnamed: 0', 'aprox_distance_shoreline', 'hours_of_month',
                 'drifter_id'], inplace=True)
df['speed'] = np.hypot(df['velocity_north'], df['velocity_east'])
df.rename(columns={'velocity_north': 'velocity_v', 'velocity_east': 'velocity_u',
                   'shore_distance_n': 'shore_distance_y', 'shore_distance_e': 'shore_distance_x',
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


#%%
cor = df.corr(numeric_only=True)
diki = cor[y_column].sort_values(ascending=True)[:-1]

import numpy as np


plt.figure(figsize=(7, 12))
plt.barh(diki.index, diki,
         capsize=5)

plt.xlabel('Correlation with grounding flag')
plt.grid(axis='x')
#plot horizontal thin lines from yticks to origin
x_lims = plt.gca().get_xlim()
for y in diki.index:
    plt.plot([x_lims[0], 0], [y, y], 'k--', alpha=0.3)

plt.xlim(x_lims)

plt.tight_layout()

plt.savefig(f'figures/corr_coef_grounding_filtered_dc1.png', dpi=300)

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

    plt.savefig(f'figures/histograms_grounding_flag_{flag}_dc1.png', dpi=300)

    plt.show()

