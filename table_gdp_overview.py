import picklemanager as pickm
import numpy as np
import matplotlib.pyplot as plt

ds = pickm.load_pickle(pickm.create_pickle_path('gdp_drogue_presence'))


#%%
print('filter\t\tnumber of observations\t\tnumber of trajectories\t\tobservations per trajectory')

print(f'Total\t{len(ds.obs)}\t{len(ds.traj)}\t{len(ds.obs)/len(ds.traj):.1f}')

n_traj_gps = ds.location_type.values.sum()
n_obs_gps = ds.rowsize.values[ds.location_type.values].sum()
print(f'GPS\t{n_obs_gps}\t{n_traj_gps}\t{n_obs_gps/n_traj_gps:.1f}')

n_traj_gps_undrogued = 0
n_traj_gps_undrogued_12km = 0
n_traj_gps_undrogued_12km_death = 0
n_obs_gps_undrogued = 0
n_obs_gps_undrogued_12km = 0
n_obs_gps_undrogued_12km_death = 0

traj_idx = np.insert(np.cumsum(ds.rowsize.values), 0, 0)
undrogue_presence = np.invert(ds.drogue_presence.values)
for j, gps, death_type in zip(ds.traj.values, ds.location_type.values, ds.type_death.values):
    if gps:
        mask_undrogued = undrogue_presence[slice(traj_idx[j], traj_idx[j + 1])]
        n_undrogued = mask_undrogued.sum()
        if n_undrogued > 0:
            n_obs_gps_undrogued += n_undrogued
            n_traj_gps_undrogued += 1

            mask_undrogue_12km = mask_undrogued & \
                                 (ds.aprox_distance_shoreline.values[slice(traj_idx[j], traj_idx[j + 1])] < 12)
            n_undrogued_12km = mask_undrogue_12km.sum()
            if n_undrogued_12km > 0:
                n_obs_gps_undrogued_12km += n_undrogued_12km
                n_traj_gps_undrogued_12km += 1
                if death_type == 1:
                    n_obs_gps_undrogued_12km_death += n_undrogued_12km
                    n_traj_gps_undrogued_12km_death += 1

print(f'& undrogued\t{n_obs_gps_undrogued}\t{n_traj_gps_undrogued}\t{n_obs_gps_undrogued/n_traj_gps_undrogued:.1f}')
print(f'& < 12km of coast\t{n_obs_gps_undrogued_12km}\t{n_traj_gps_undrogued_12km}\t'
        f'{n_obs_gps_undrogued_12km/n_traj_gps_undrogued_12km:.1f}')
print(f'& death type 1\t{n_obs_gps_undrogued_12km_death}\t{n_traj_gps_undrogued_12km_death}\t'
        f'{n_obs_gps_undrogued_12km_death/n_traj_gps_undrogued_12km_death:.1f}')


#%%

death_types = np.sort(np.unique(ds.type_death.values))
death_type_count = np.zeros(len(death_types))
for death_type in death_types:
    death_type_count[death_type] = np.sum(ds.type_death.values == death_type)

death_colors = {0: 'r', 1: 'g', 2: 'b', 3: 'y', 4: 'c', 5: 'm', 6: 'aquamarine'}
fig, ax = plt.subplots()
ax.bar(death_types, death_type_count, color=death_colors.values())
ax.set_ylabel('Fraction')
ax.set_xlabel('Death Type')
plt.show()

