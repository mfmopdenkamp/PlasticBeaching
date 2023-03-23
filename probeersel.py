
import pandas as pd
import numpy as np

df = pd.read_csv('data/events_wind.csv')


def split_events(end_obs_indexes, start_obs_indexes, length_threshold):
    split_obs_indexes_to_insert = np.array([], dtype=int)
    where_to_insert_event_indexes = np.array([], dtype=int)

    event_lengths = end_obs_indexes - start_obs_indexes
    split_length = int(length_threshold / 2)
    for i_event, event_length in enumerate(event_lengths):
        if event_length >= length_threshold:
            event_split_obs_indexes = np.arange(split_length, event_length - split_length + 1, split_length) \
                                      + start_obs_indexes[i_event]  # start counting from start event instead of zero
            split_obs_indexes_to_insert = np.append(split_obs_indexes_to_insert, event_split_obs_indexes)
            where_to_insert_event_indexes = np.append(where_to_insert_event_indexes,
                                                      np.ones(len(event_split_obs_indexes), dtype=int)
                                                      * i_event)

    # insert new events
    start_obs_indexes = np.insert(start_obs_indexes, where_to_insert_event_indexes + 1, split_obs_indexes_to_insert)
    end_obs_indexes = np.insert(end_obs_indexes, where_to_insert_event_indexes, split_obs_indexes_to_insert)

    return start_obs_indexes, end_obs_indexes

s = np.array([1,8,44,58])
e = np.array([5,11,55,60])
t = 4

s,e = split_events(e,s,t)

