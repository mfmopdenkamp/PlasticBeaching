import numpy as np
import picklemanager as pickm
import pandas as pd
import toolbox as tb
import load_data
import plotter

# %% Settings
plot_things = False


ds_gdp = pickm.load_pickle('ds_gdp_drogue_presence')
ds_gdp = load_data.load_subset(type_death=1, ds=ds_gdp)


# %% Create segments
print(f'Creating segments...', end='')


def get_segments(ds, max_seg_len_h):
    """Split the trajectories into segments of a maximum length. The algorithm takes into account the fact that the
    trajectories are not continuous in time, because of the filtering on approximate distance to the shoreline."""
    start_segment = []
    end_segment = []
    beaching_flags = []

    return np.array(start_segment), np.array(end_segment)


segment_length_h = 24
start_index, end_index = get_segments(ds_gdp, segment_length_h)

# check if all segments have a duration of segment_length_h
segments_durations = (ds_gdp.time.values[end_index - 1] - ds_gdp.time.values[start_index]) / np.timedelta64(1, 'h')
if (segments_durations != 24).any():
    raise ValueError('Some segments have a duration not equal to 24 hours!')

print(f'Number of {segment_length_h}h segments :', len(start_index))


# %% Get beaching flags and beaching observations
print(f'Getting beaching flags and beaching observations...', end='')


def get_beaching_flags(ds, s_obs, e_obs):
    if len(s_obs) != len(e_obs):
        raise ValueError('"starts_obs" and "end_obs" must have equal lengths!')

    b_flags = np.zeros(len(s_obs), dtype=bool)
    mask_drifter_on_shore = np.zeros(len(ds.obs), dtype=bool)

    for traj in ds.traj.values:
        obs = tb.obs_from_traj(ds, traj)
        mask_drifter_on_shore[obs] = tb.get_obs_drifter_on_shore(ds.isel(obs=obs, traj=traj))

    for index, (i_s, i_e) in enumerate(zip(s_obs, e_obs)):
        if mask_drifter_on_shore[i_s:i_e].any():
            b_flags[index] = True
    return b_flags, mask_drifter_on_shore


beaching_flags, beaching_mask = get_beaching_flags(ds_gdp, start_index, end_index)

print('Number of beaching events:', beaching_flags.sum())

# %% Delete segments that start with beaching observations

mask_start_beaching = np.zeros(len(start_index), dtype=bool)

for i, start_ob in zip(np.where(beaching_flags)[0], start_index[beaching_flags]):
    if beaching_mask[start_ob]:
        mask_start_beaching[i] = True

start_index = start_index[~mask_start_beaching]
end_index = end_index[~mask_start_beaching]
beaching_flags = beaching_flags[~mask_start_beaching]

print('Number of deleted segments because they start with beaching: ', mask_start_beaching.sum())


# %% Create initial supervised dataframe

df = pd.DataFrame(data={'beaching_flag': beaching_flags,
                        'time_start': ds_gdp.time[start_index],
                        'time_end': ds_gdp.time[end_index - 1],
                        'drifter_id': ds_gdp.ids[start_index],
                        'latitude_start': ds_gdp.latitude[start_index],
                        'latitude_end': ds_gdp.latitude[end_index - 1],
                        'longitude_start': ds_gdp.longitude[start_index],
                        'longitude_end': ds_gdp.longitude[end_index - 1],
                        'velocity_east': ds_gdp.ve[start_index],  # TODO: change to mean of previous segment_length
                        'velocity_north': ds_gdp.vn[start_index]  # TODO: change to mean of previous segment_length
                        })

# Make sure the time is in datetime format
df['time_start'] = pd.to_datetime(df['time_start'])
df['time_end'] = pd.to_datetime(df['time_end'])

# sort the segments dataframe by time
df.sort_values('time_start', inplace=True)

# Save the dataframe to csv
df.to_csv(f'data/df_supervised_{segment_length_h}h_{name}.csv', index_label='ID')
print(f'Saved dataframe to data/df_supervised_{segment_length_h}h_{name}.csv')

# %% Plotting
if plot_things:
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    def plot_beaching_trajectories(ds, ax=None, s=15, ds_beaching_obs=None, df_shore=pd.DataFrame()):
        """given a dataset, plot the trajectories on a map"""
        if ax is None:
            plt.figure(figsize=(12, 8), dpi=300)
            ax = plt.axes(projection=ccrs.PlateCarree())
            extent_offset = 0.2
            ax.set_xlim([ds['longitude'].min() - extent_offset, ds['longitude'].max() + extent_offset])
            ax.set_ylim([ds['latitude'].min() - extent_offset, ds['latitude'].max() + extent_offset])

        ax.scatter(ds.longitude, ds.latitude, transform=ccrs.PlateCarree(), s=s, c='midnightblue', alpha=0.5)
        ax.plot(ds.longitude, ds.latitude, ':k', transform=ccrs.PlateCarree(), alpha=0.5)

        if ds_beaching_obs is not None:
            ax.scatter(ds_beaching_obs.longitude, ds_beaching_obs.latitude, transform=ccrs.PlateCarree(), s=s * 2,
                       c='r')

        if not df_shore.empty:
            df_shore.plot(ax=ax, color='b')
        # else:
        #     ax.coastlines()

        plt.tight_layout()
        plt.show()


    beaching_event_obs = []
    for i_b in np.where(beaching_flags)[0]:
        beaching_event_obs.append([i for i in range(start_index[i_b], end_index[i_b])])

    extent_offset = 0.1
    for i in range(len(beaching_event_obs)):
        ds_select = ds_gdp.isel(obs=beaching_event_obs[i])

        extent = (ds_select['longitude'].min() - extent_offset, ds_select['longitude'].max() + extent_offset,
                  ds_select['latitude'].min() - extent_offset, ds_select['latitude'].max() + extent_offset)

        fig, ax = plotter.get_sophie_subplots(figsize=None, extent=extent)
        plot_beaching_trajectories(ds_select, ax, s=12, ds_beaching_obs=ds_gdp.isel(obs=beaching_obs_list[i]))
