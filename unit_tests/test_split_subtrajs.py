import numpy as np



def split_subtrajs(start_obs, end_obs, beaching_flags, beaching_obs_list, split_length_h=24):
    # Split subtrajs based on time. Use index for this, since they correspond to exactly 1 hour.
    # New subtrajs may not be smaller than the length threshold!
    split_obs_to_insert = np.array([], dtype=int)
    beaching_flags_to_insert = np.array([], dtype=bool)
    where_to_insert_new_subtraj = np.array([], dtype=int)
    where_to_change_beaching_flags = np.array([], dtype=int)
    subtraj_lengths = end_obs - start_obs

    i_beaching_event = 0
    for i_subtraj, (start_ob, end_ob, subtraj_length, beaching_flag) in enumerate(zip(start_obs, end_obs,
                                                                                      subtraj_lengths, beaching_flags)):

        # determine split points of subtrajs based on their length
        if subtraj_length >= split_length_h * 2:
            subtraj_split_obs = np.arange(split_length_h, subtraj_length - split_length_h + 1, split_length_h) \
                                + start_ob  # start counting from start subtraj instead of zero

            # if beaching took place, check new beaching flags
            if beaching_flag:

                new_beaching_flags_from_subtraj = np.array(np.zeros(len(subtraj_split_obs)), dtype=bool)

                # check if original beaching flag must be changed to False
                if not np.any(np.in1d(np.arange(start_ob, subtraj_split_obs[0]), beaching_obs_list[i_beaching_event])):
                    where_to_change_beaching_flags = np.append(where_to_change_beaching_flags, i_subtraj)

                for j in range(len(subtraj_split_obs) - 1):
                    obs_in_new_sub_traj = np.arange(subtraj_split_obs[j], subtraj_split_obs[j+1])
                    if np.any(np.in1d(obs_in_new_sub_traj, beaching_obs_list[i_beaching_event])):
                        new_beaching_flags_from_subtraj[j] = True

                # check if the last part of the subtraj is beaching
                if np.any(np.in1d(np.arange(subtraj_split_obs[-1], end_ob), beaching_obs_list[i_beaching_event])):
                    new_beaching_flags_from_subtraj[-1] = True

                i_beaching_event += 1

            # if no beaching took place, set all new beaching flags to False
            else:
                new_beaching_flags_from_subtraj = np.zeros(len(subtraj_split_obs), dtype=bool)

            # append new split points for this coordinate
            split_obs_to_insert = np.append(split_obs_to_insert, subtraj_split_obs)
            beaching_flags_to_insert = np.append(beaching_flags_to_insert, new_beaching_flags_from_subtraj)
            where_to_insert_new_subtraj = np.append(where_to_insert_new_subtraj,
                                                    np.ones(len(subtraj_split_obs), dtype=int) * i_subtraj)

    # change beaching flags
    beaching_flags[where_to_change_beaching_flags] = False

    # insert new subtrajs
    start_obs = np.insert(start_obs, where_to_insert_new_subtraj + 1, split_obs_to_insert)
    end_obs = np.insert(end_obs, where_to_insert_new_subtraj, split_obs_to_insert)
    beaching_flags = np.insert(beaching_flags, where_to_insert_new_subtraj + 1, beaching_flags_to_insert)

    return start_obs, end_obs, beaching_flags

beaching_obs_list = [np.array([11,12,14,15]), np.array([34,35,36,37]), np.array([50,51,52,53])]
start_obs, end_obs, beaching_flags = split_subtrajs(np.array([8, 30, 40, 77]), np.array([17, 37, 60, 88]),
                                                    np.array([1, 1, 1, 0]), beaching_obs_list, split_length_h=3)


#%% Delete subtrajs that start with beaching obs
all_beaching_obs = np.concatenate(beaching_obs_list)
mask_start_beaching = np.in1d(start_obs, all_beaching_obs)
start_obs = start_obs[~mask_start_beaching]
end_obs = end_obs[~mask_start_beaching]
beaching_flags = beaching_flags[~mask_start_beaching]

import pandas as pd
df = pd.DataFrame({'start_obs': start_obs, 'end_obs': end_obs, 'beaching_flags': beaching_flags})