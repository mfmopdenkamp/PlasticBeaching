from file_names import file_name_4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv(file_name_4, parse_dates=['time_start', 'time_end'], index_col='ID')

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

#%%
df['score_shore_360deg_all'] = df.score_shore_360deg_10km + df.score_shore_360deg_8km + df.score_shore_360deg_5km + df.score_shore_360deg_2km
# normalize
df.score_shore_360deg_all = df.score_shore_360deg_all / df.score_shore_360deg_all.max()
df.score_shore_360deg_10km = df.score_shore_360deg_10km / df.score_shore_360deg_10km.max()
df.score_shore_360deg_8km = df.score_shore_360deg_8km / df.score_shore_360deg_8km.max()
df.score_shore_360deg_5km = df.score_shore_360deg_5km / df.score_shore_360deg_5km.max()
df.score_shore_360deg_2km = df.score_shore_360deg_2km / df.score_shore_360deg_2km.max()


shore_score_thresholds = np.arange(0, 1.1, 0.02)
beaching_prob_by_shore_score_360deg_all = np.zeros(len(shore_score_thresholds))
beaching_prob_by_shore_score_360deg_10km = np.zeros(len(shore_score_thresholds))
beaching_prob_by_shore_score_360deg_8km = np.zeros(len(shore_score_thresholds))
beaching_prob_by_shore_score_360deg_5km = np.zeros(len(shore_score_thresholds))
beaching_prob_by_shore_score_360deg_2km = np.zeros(len(shore_score_thresholds))

for i, shore_score_threshold in enumerate(shore_score_thresholds):
    df_filtered = df[df.score_shore_360deg_all >= shore_score_threshold]
    beaching_prob_by_shore_score_360deg_all[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_shore_360deg_10km >= shore_score_threshold]
    beaching_prob_by_shore_score_360deg_10km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_shore_360deg_8km >= shore_score_threshold]
    beaching_prob_by_shore_score_360deg_8km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_shore_360deg_5km >= shore_score_threshold]
    beaching_prob_by_shore_score_360deg_5km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_shore_360deg_2km >= shore_score_threshold]
    beaching_prob_by_shore_score_360deg_2km[i] = df_filtered.beaching_flag.mean()


plt.figure(figsize=(7, 5))
plt.plot(shore_score_thresholds, beaching_prob_by_shore_score_360deg_all, 'ko-', label='all')
plt.plot(shore_score_thresholds, beaching_prob_by_shore_score_360deg_10km, 'o-', label='10km')
plt.plot(shore_score_thresholds, beaching_prob_by_shore_score_360deg_8km, 'o-', label='8km')
plt.plot(shore_score_thresholds, beaching_prob_by_shore_score_360deg_5km, 'o-', label='5km')
plt.plot(shore_score_thresholds, beaching_prob_by_shore_score_360deg_2km, 'o-', label='2km')
plt.legend(title='radius')
plt.xlabel('normalized score threshold')
plt.ylabel('beaching probability')
plt.title('shore scores 360deg')
plt.grid()
plt.savefig('figures/beaching_probability_vs_shore_score_threshold_360deg_all.png', dpi=300)
plt.show()

#%%
df['score_shore_wind_all'] = df.score_shore_22deg_14km + df.score_shore_45deg_10km + df.score_shore_60deg_9km +\
                             df.score_shore_180deg_5km + df.score_shore_270deg_4km
# normalize
df.score_shore_wind_all = df.score_shore_wind_all / df.score_shore_wind_all.max()
df.score_shore_22deg_14km = df.score_shore_22deg_14km / df.score_shore_22deg_14km.max()
df.score_shore_45deg_10km = df.score_shore_45deg_10km / df.score_shore_45deg_10km.max()
df.score_shore_60deg_9km = df.score_shore_60deg_9km / df.score_shore_60deg_9km.max()
df.score_shore_180deg_5km = df.score_shore_180deg_5km / df.score_shore_180deg_5km.max()
df.score_shore_270deg_4km = df.score_shore_270deg_4km / df.score_shore_270deg_4km.max()

beaching_prob_by_shore_score_wind_all = np.zeros(len(shore_score_thresholds))
beaching_prob_by_shore_score_22deg_14km = np.zeros(len(shore_score_thresholds))
beaching_prob_by_shore_score_45deg_10km = np.zeros(len(shore_score_thresholds))
beaching_prob_by_shore_score_60deg_9km = np.zeros(len(shore_score_thresholds))
beaching_prob_by_shore_score_180deg_5km = np.zeros(len(shore_score_thresholds))
beaching_prob_by_shore_score_270deg_4km = np.zeros(len(shore_score_thresholds))

for i, shore_score_threshold in enumerate(shore_score_thresholds):
    df_filtered = df[df.score_shore_wind_all >= shore_score_threshold]
    beaching_prob_by_shore_score_wind_all[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_shore_22deg_14km >= shore_score_threshold]
    beaching_prob_by_shore_score_22deg_14km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_shore_45deg_10km >= shore_score_threshold]
    beaching_prob_by_shore_score_45deg_10km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_shore_60deg_9km >= shore_score_threshold]
    beaching_prob_by_shore_score_60deg_9km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_shore_180deg_5km >= shore_score_threshold]
    beaching_prob_by_shore_score_180deg_5km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_shore_270deg_4km >= shore_score_threshold]
    beaching_prob_by_shore_score_270deg_4km[i] = df_filtered.beaching_flag.mean()


plt.figure(figsize=(7, 5))
plt.plot(shore_score_thresholds, beaching_prob_by_shore_score_wind_all, 'ko-', label='all')
plt.plot(shore_score_thresholds, beaching_prob_by_shore_score_22deg_14km, 'o-', label='22deg_14km')
plt.plot(shore_score_thresholds, beaching_prob_by_shore_score_45deg_10km, 'o-', label='45deg_10km')
plt.plot(shore_score_thresholds, beaching_prob_by_shore_score_60deg_9km, 'o-', label='60deg_9km')
plt.plot(shore_score_thresholds, beaching_prob_by_shore_score_180deg_5km, 'o-', label='180deg_5km')
plt.plot(shore_score_thresholds, beaching_prob_by_shore_score_270deg_4km, 'o-', c='cyan', label='270deg_4km')
plt.legend(title='radius')
plt.xlabel('normalized score threshold')
plt.ylabel('beaching probability')
plt.title('shore scores wind')
plt.grid()
plt.savefig('figures/beaching_probability_vs_shore_score_threshold_wind_all.png', dpi=300)
plt.show()

#%% For beach only
df['score_beach_360deg_all'] = df.score_beach_360deg_10km + df.score_beach_360deg_8km + df.score_beach_360deg_5km + df.score_beach_360deg_2km
# normalize
df.score_beach_360deg_all = df.score_beach_360deg_all / df.score_beach_360deg_all.max()
df.score_beach_360deg_10km = df.score_beach_360deg_10km / df.score_beach_360deg_10km.max()
df.score_beach_360deg_8km = df.score_beach_360deg_8km / df.score_beach_360deg_8km.max()
df.score_beach_360deg_5km = df.score_beach_360deg_5km / df.score_beach_360deg_5km.max()
df.score_beach_360deg_2km = df.score_beach_360deg_2km / df.score_beach_360deg_2km.max()


beach_score_thresholds = np.arange(0, 1.1, 0.02)
beaching_prob_by_beach_score_360deg_all = np.zeros(len(beach_score_thresholds))
beaching_prob_by_beach_score_360deg_10km = np.zeros(len(beach_score_thresholds))
beaching_prob_by_beach_score_360deg_8km = np.zeros(len(beach_score_thresholds))
beaching_prob_by_beach_score_360deg_5km = np.zeros(len(beach_score_thresholds))
beaching_prob_by_beach_score_360deg_2km = np.zeros(len(beach_score_thresholds))

for i, beach_score_threshold in enumerate(beach_score_thresholds):
    df_filtered = df[df.score_beach_360deg_all >= beach_score_threshold]
    beaching_prob_by_beach_score_360deg_all[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_beach_360deg_10km >= beach_score_threshold]
    beaching_prob_by_beach_score_360deg_10km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_beach_360deg_8km >= beach_score_threshold]
    beaching_prob_by_beach_score_360deg_8km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_beach_360deg_5km >= beach_score_threshold]
    beaching_prob_by_beach_score_360deg_5km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_beach_360deg_2km >= beach_score_threshold]
    beaching_prob_by_beach_score_360deg_2km[i] = df_filtered.beaching_flag.mean()


plt.figure(figsize=(7, 5))
plt.plot(beach_score_thresholds, beaching_prob_by_beach_score_360deg_all, 'ko-', label='all')
plt.plot(beach_score_thresholds, beaching_prob_by_beach_score_360deg_10km, 'o-', label='10km')
plt.plot(beach_score_thresholds, beaching_prob_by_beach_score_360deg_8km, 'o-', label='8km')
plt.plot(beach_score_thresholds, beaching_prob_by_beach_score_360deg_5km, 'o-', label='5km')
plt.plot(beach_score_thresholds, beaching_prob_by_beach_score_360deg_2km, 'o-', label='2km')
plt.legend(title='radius')
plt.xlabel('normalized score threshold')
plt.ylabel('beaching probability')
plt.title('beach scores 360deg')
plt.grid()
plt.savefig('figures/beaching_probability_vs_beach_score_threshold_360deg_all.png', dpi=300)
plt.show()

df['score_beach_wind_all'] = df.score_beach_22deg_14km + df.score_beach_45deg_10km + df.score_beach_60deg_9km +\
                             df.score_beach_180deg_5km + df.score_beach_270deg_4km
# normalize
df.score_beach_wind_all = df.score_beach_wind_all / df.score_beach_wind_all.max()
df.score_beach_22deg_14km = df.score_beach_22deg_14km / df.score_beach_22deg_14km.max()
df.score_beach_45deg_10km = df.score_beach_45deg_10km / df.score_beach_45deg_10km.max()
df.score_beach_60deg_9km = df.score_beach_60deg_9km / df.score_beach_60deg_9km.max()
df.score_beach_180deg_5km = df.score_beach_180deg_5km / df.score_beach_180deg_5km.max()
df.score_beach_270deg_4km = df.score_beach_270deg_4km / df.score_beach_270deg_4km.max()

beaching_prob_by_beach_score_wind_all = np.zeros(len(beach_score_thresholds))
beaching_prob_by_beach_score_22deg_14km = np.zeros(len(beach_score_thresholds))
beaching_prob_by_beach_score_45deg_10km = np.zeros(len(beach_score_thresholds))
beaching_prob_by_beach_score_60deg_9km = np.zeros(len(beach_score_thresholds))
beaching_prob_by_beach_score_180deg_5km = np.zeros(len(beach_score_thresholds))
beaching_prob_by_beach_score_270deg_4km = np.zeros(len(beach_score_thresholds))

for i, beach_score_threshold in enumerate(beach_score_thresholds):
    df_filtered = df[df.score_beach_wind_all >= beach_score_threshold]
    beaching_prob_by_beach_score_wind_all[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_beach_22deg_14km >= beach_score_threshold]
    beaching_prob_by_beach_score_22deg_14km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_beach_45deg_10km >= beach_score_threshold]
    beaching_prob_by_beach_score_45deg_10km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_beach_60deg_9km >= beach_score_threshold]
    beaching_prob_by_beach_score_60deg_9km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_beach_180deg_5km >= beach_score_threshold]
    beaching_prob_by_beach_score_180deg_5km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_beach_270deg_4km >= beach_score_threshold]
    beaching_prob_by_beach_score_270deg_4km[i] = df_filtered.beaching_flag.mean()


plt.figure(figsize=(7, 5))
plt.plot(beach_score_thresholds, beaching_prob_by_beach_score_wind_all, 'ko-', label='all')
plt.plot(beach_score_thresholds, beaching_prob_by_beach_score_22deg_14km, 'o-', label='22deg_14km')
plt.plot(beach_score_thresholds, beaching_prob_by_beach_score_45deg_10km, 'o-', label='45deg_10km')
plt.plot(beach_score_thresholds, beaching_prob_by_beach_score_60deg_9km, 'o-', label='60deg_9km')
plt.plot(beach_score_thresholds, beaching_prob_by_beach_score_180deg_5km, 'o-', label='180deg_5km')
plt.plot(beach_score_thresholds, beaching_prob_by_beach_score_270deg_4km, 'o-', c='cyan', label='270deg_4km')
plt.legend(title='radius')
plt.xlabel('normalized score threshold')
plt.ylabel('beaching probability')
plt.title('beach scores wind')
plt.grid()
plt.savefig('figures/beaching_probability_vs_beach_score_threshold_wind_all.png', dpi=300)
plt.show()


#%% bedrock
df['score_bedrock_360deg_all'] = df.score_bedrock_360deg_10km + df.score_bedrock_360deg_8km + df.score_bedrock_360deg_5km + df.score_bedrock_360deg_2km
# normalize
df.score_bedrock_360deg_all = df.score_bedrock_360deg_all / df.score_bedrock_360deg_all.max()
df.score_bedrock_360deg_10km = df.score_bedrock_360deg_10km / df.score_bedrock_360deg_10km.max()
df.score_bedrock_360deg_8km = df.score_bedrock_360deg_8km / df.score_bedrock_360deg_8km.max()
df.score_bedrock_360deg_5km = df.score_bedrock_360deg_5km / df.score_bedrock_360deg_5km.max()
df.score_bedrock_360deg_2km = df.score_bedrock_360deg_2km / df.score_bedrock_360deg_2km.max()


bedrock_score_thresholds = np.arange(0, 1.1, 0.02)
beaching_prob_by_bedrock_score_360deg_all = np.zeros(len(bedrock_score_thresholds))
beaching_prob_by_bedrock_score_360deg_10km = np.zeros(len(bedrock_score_thresholds))
beaching_prob_by_bedrock_score_360deg_8km = np.zeros(len(bedrock_score_thresholds))
beaching_prob_by_bedrock_score_360deg_5km = np.zeros(len(bedrock_score_thresholds))
beaching_prob_by_bedrock_score_360deg_2km = np.zeros(len(bedrock_score_thresholds))

for i, bedrock_score_threshold in enumerate(bedrock_score_thresholds):
    df_filtered = df[df.score_bedrock_360deg_all >= bedrock_score_threshold]
    beaching_prob_by_bedrock_score_360deg_all[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_bedrock_360deg_10km >= bedrock_score_threshold]
    beaching_prob_by_bedrock_score_360deg_10km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_bedrock_360deg_8km >= bedrock_score_threshold]
    beaching_prob_by_bedrock_score_360deg_8km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_bedrock_360deg_5km >= bedrock_score_threshold]
    beaching_prob_by_bedrock_score_360deg_5km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_bedrock_360deg_2km >= bedrock_score_threshold]
    beaching_prob_by_bedrock_score_360deg_2km[i] = df_filtered.beaching_flag.mean()


plt.figure(figsize=(7, 5))
plt.plot(bedrock_score_thresholds, beaching_prob_by_bedrock_score_360deg_all, 'ko-', label='all')
plt.plot(bedrock_score_thresholds, beaching_prob_by_bedrock_score_360deg_10km, 'o-', label='10km')
plt.plot(bedrock_score_thresholds, beaching_prob_by_bedrock_score_360deg_8km, 'o-', label='8km')
plt.plot(bedrock_score_thresholds, beaching_prob_by_bedrock_score_360deg_5km, 'o-', label='5km')
plt.plot(bedrock_score_thresholds, beaching_prob_by_bedrock_score_360deg_2km, 'o-', label='2km')
plt.legend(title='radius')
plt.xlabel('normalized score threshold')
plt.ylabel('beaching probability')
plt.title('bedrock scores 360deg')
plt.grid()
plt.savefig('figures/beaching_probability_vs_bedrock_score_threshold_360deg_all.png', dpi=300)
plt.show()

df['score_bedrock_wind_all'] = df.score_bedrock_22deg_14km + df.score_bedrock_45deg_10km + df.score_bedrock_60deg_9km +\
                             df.score_bedrock_180deg_5km + df.score_bedrock_270deg_4km
# normalize
df.score_bedrock_wind_all = df.score_bedrock_wind_all / df.score_bedrock_wind_all.max()
df.score_bedrock_22deg_14km = df.score_bedrock_22deg_14km / df.score_bedrock_22deg_14km.max()
df.score_bedrock_45deg_10km = df.score_bedrock_45deg_10km / df.score_bedrock_45deg_10km.max()
df.score_bedrock_60deg_9km = df.score_bedrock_60deg_9km / df.score_bedrock_60deg_9km.max()
df.score_bedrock_180deg_5km = df.score_bedrock_180deg_5km / df.score_bedrock_180deg_5km.max()
df.score_bedrock_270deg_4km = df.score_bedrock_270deg_4km / df.score_bedrock_270deg_4km.max()

beaching_prob_by_bedrock_score_wind_all = np.zeros(len(bedrock_score_thresholds))
beaching_prob_by_bedrock_score_22deg_14km = np.zeros(len(bedrock_score_thresholds))
beaching_prob_by_bedrock_score_45deg_10km = np.zeros(len(bedrock_score_thresholds))
beaching_prob_by_bedrock_score_60deg_9km = np.zeros(len(bedrock_score_thresholds))
beaching_prob_by_bedrock_score_180deg_5km = np.zeros(len(bedrock_score_thresholds))
beaching_prob_by_bedrock_score_270deg_4km = np.zeros(len(bedrock_score_thresholds))

for i, bedrock_score_threshold in enumerate(bedrock_score_thresholds):
    df_filtered = df[df.score_bedrock_wind_all >= bedrock_score_threshold]
    beaching_prob_by_bedrock_score_wind_all[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_bedrock_22deg_14km >= bedrock_score_threshold]
    beaching_prob_by_bedrock_score_22deg_14km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_bedrock_45deg_10km >= bedrock_score_threshold]
    beaching_prob_by_bedrock_score_45deg_10km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_bedrock_60deg_9km >= bedrock_score_threshold]
    beaching_prob_by_bedrock_score_60deg_9km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_bedrock_180deg_5km >= bedrock_score_threshold]
    beaching_prob_by_bedrock_score_180deg_5km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_bedrock_270deg_4km >= bedrock_score_threshold]
    beaching_prob_by_bedrock_score_270deg_4km[i] = df_filtered.beaching_flag.mean()


plt.figure(figsize=(7, 5))
plt.plot(bedrock_score_thresholds, beaching_prob_by_bedrock_score_wind_all, 'ko-', label='all')
plt.plot(bedrock_score_thresholds, beaching_prob_by_bedrock_score_22deg_14km, 'o-', label='22deg_14km')
plt.plot(bedrock_score_thresholds, beaching_prob_by_bedrock_score_45deg_10km, 'o-', label='45deg_10km')
plt.plot(bedrock_score_thresholds, beaching_prob_by_bedrock_score_60deg_9km, 'o-', label='60deg_9km')
plt.plot(bedrock_score_thresholds, beaching_prob_by_bedrock_score_180deg_5km, 'o-', label='180deg_5km')
plt.plot(bedrock_score_thresholds, beaching_prob_by_bedrock_score_270deg_4km, 'o-', c='cyan', label='270deg_4km')
plt.legend(title='radius')
plt.xlabel('normalized score threshold')
plt.ylabel('beaching probability')
plt.title('bedrock scores wind')
plt.grid()
plt.savefig('figures/beaching_probability_vs_bedrock_score_threshold_wind_all.png', dpi=300)
plt.show()

#%% wetland
df['score_wetland_360deg_all'] = df.score_wetland_360deg_10km + df.score_wetland_360deg_8km + df.score_wetland_360deg_5km + df.score_wetland_360deg_2km
# normalize
df.score_wetland_360deg_all = df.score_wetland_360deg_all / df.score_wetland_360deg_all.max()
df.score_wetland_360deg_10km = df.score_wetland_360deg_10km / df.score_wetland_360deg_10km.max()
df.score_wetland_360deg_8km = df.score_wetland_360deg_8km / df.score_wetland_360deg_8km.max()
df.score_wetland_360deg_5km = df.score_wetland_360deg_5km / df.score_wetland_360deg_5km.max()
df.score_wetland_360deg_2km = df.score_wetland_360deg_2km / df.score_wetland_360deg_2km.max()


wetland_score_thresholds = np.arange(0, 1.1, 0.02)
beaching_prob_by_wetland_score_360deg_all = np.zeros(len(wetland_score_thresholds))
beaching_prob_by_wetland_score_360deg_10km = np.zeros(len(wetland_score_thresholds))
beaching_prob_by_wetland_score_360deg_8km = np.zeros(len(wetland_score_thresholds))
beaching_prob_by_wetland_score_360deg_5km = np.zeros(len(wetland_score_thresholds))
beaching_prob_by_wetland_score_360deg_2km = np.zeros(len(wetland_score_thresholds))

for i, wetland_score_threshold in enumerate(wetland_score_thresholds):
    df_filtered = df[df.score_wetland_360deg_all >= wetland_score_threshold]
    beaching_prob_by_wetland_score_360deg_all[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_wetland_360deg_10km >= wetland_score_threshold]
    beaching_prob_by_wetland_score_360deg_10km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_wetland_360deg_8km >= wetland_score_threshold]
    beaching_prob_by_wetland_score_360deg_8km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_wetland_360deg_5km >= wetland_score_threshold]
    beaching_prob_by_wetland_score_360deg_5km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_wetland_360deg_2km >= wetland_score_threshold]
    beaching_prob_by_wetland_score_360deg_2km[i] = df_filtered.beaching_flag.mean()


plt.figure(figsize=(7, 5))
plt.plot(wetland_score_thresholds, beaching_prob_by_wetland_score_360deg_all, 'ko-', label='all')
plt.plot(wetland_score_thresholds, beaching_prob_by_wetland_score_360deg_10km, 'o-', label='10km')
plt.plot(wetland_score_thresholds, beaching_prob_by_wetland_score_360deg_8km, 'o-', label='8km')
plt.plot(wetland_score_thresholds, beaching_prob_by_wetland_score_360deg_5km, 'o-', label='5km')
plt.plot(wetland_score_thresholds, beaching_prob_by_wetland_score_360deg_2km, 'o-', label='2km')
plt.legend(title='radius')
plt.xlabel('normalized score threshold')
plt.ylabel('beaching probability')
plt.title('wetland scores 360deg')
plt.grid()
plt.savefig('figures/beaching_probability_vs_wetland_score_threshold_360deg_all.png', dpi=300)
plt.show()

df['score_wetland_wind_all'] = df.score_wetland_22deg_14km + df.score_wetland_45deg_10km + df.score_wetland_60deg_9km +\
                             df.score_wetland_180deg_5km + df.score_wetland_270deg_4km
# normalize
df.score_wetland_wind_all = df.score_wetland_wind_all / df.score_wetland_wind_all.max()
df.score_wetland_22deg_14km = df.score_wetland_22deg_14km / df.score_wetland_22deg_14km.max()
df.score_wetland_45deg_10km = df.score_wetland_45deg_10km / df.score_wetland_45deg_10km.max()
df.score_wetland_60deg_9km = df.score_wetland_60deg_9km / df.score_wetland_60deg_9km.max()
df.score_wetland_180deg_5km = df.score_wetland_180deg_5km / df.score_wetland_180deg_5km.max()
df.score_wetland_270deg_4km = df.score_wetland_270deg_4km / df.score_wetland_270deg_4km.max()

beaching_prob_by_wetland_score_wind_all = np.zeros(len(wetland_score_thresholds))
beaching_prob_by_wetland_score_22deg_14km = np.zeros(len(wetland_score_thresholds))
beaching_prob_by_wetland_score_45deg_10km = np.zeros(len(wetland_score_thresholds))
beaching_prob_by_wetland_score_60deg_9km = np.zeros(len(wetland_score_thresholds))
beaching_prob_by_wetland_score_180deg_5km = np.zeros(len(wetland_score_thresholds))
beaching_prob_by_wetland_score_270deg_4km = np.zeros(len(wetland_score_thresholds))

for i, wetland_score_threshold in enumerate(wetland_score_thresholds):
    df_filtered = df[df.score_wetland_wind_all >= wetland_score_threshold]
    beaching_prob_by_wetland_score_wind_all[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_wetland_22deg_14km >= wetland_score_threshold]
    beaching_prob_by_wetland_score_22deg_14km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_wetland_45deg_10km >= wetland_score_threshold]
    beaching_prob_by_wetland_score_45deg_10km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_wetland_60deg_9km >= wetland_score_threshold]
    beaching_prob_by_wetland_score_60deg_9km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_wetland_180deg_5km >= wetland_score_threshold]
    beaching_prob_by_wetland_score_180deg_5km[i] = df_filtered.beaching_flag.mean()
    df_filtered = df[df.score_wetland_270deg_4km >= wetland_score_threshold]
    beaching_prob_by_wetland_score_270deg_4km[i] = df_filtered.beaching_flag.mean()


plt.figure(figsize=(7, 5))
plt.plot(wetland_score_thresholds, beaching_prob_by_wetland_score_wind_all, 'ko-', label='all')
plt.plot(wetland_score_thresholds, beaching_prob_by_wetland_score_22deg_14km, 'o-', label='22deg_14km')
plt.plot(wetland_score_thresholds, beaching_prob_by_wetland_score_45deg_10km, 'o-', label='45deg_10km')
plt.plot(wetland_score_thresholds, beaching_prob_by_wetland_score_60deg_9km, 'o-', label='60deg_9km')
plt.plot(wetland_score_thresholds, beaching_prob_by_wetland_score_180deg_5km, 'o-', label='180deg_5km')
plt.plot(wetland_score_thresholds, beaching_prob_by_wetland_score_270deg_4km, 'o-', c='cyan', label='270deg_4km')
plt.legend(title='radius')
plt.xlabel('normalized score threshold')
plt.ylabel('beaching probability')
plt.title('wetland scores wind')
plt.grid()
plt.savefig('figures/beaching_probability_vs_wetland_score_threshold_wind_all.png', dpi=300)
plt.show()