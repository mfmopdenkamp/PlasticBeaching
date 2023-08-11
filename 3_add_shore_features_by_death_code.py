import numpy as np
from numba import njit, prange
import pandas as pd
import math
import picklemanager as pm
from file_names import *

df_states_0 = pd.read_csv(file_name_2, parse_dates=['time'], infer_datetime_format=True)

df_states = df_states_0[df_states_0['aprox_distance_shoreline'] < 12]  # Only use states that are within 12 km of the shore

df_shore = pm.load_pickle(pm.create_pickle_path('df_shoreline_f_lonlat'))
df_cm = pm.load_pickle(pm.create_pickle_path('df_coastal_morphology_lonlat'))

#%% Initialize output

output_dtype = np.float32
init_distance = np.finfo(output_dtype).max

base_alpha = 45
base_distance = 10000
alphas = 360 / np.array([16, 8, 6, 4, 2, 4 / 3])
distances = base_distance * np.sqrt(base_alpha / alphas)
alphas = np.append(alphas, np.array([360, 360, 360, 360]))
distances = np.append(distances, np.ones(4) * base_distance * [1, 3/4, 1/2, 1/4])

area_labels = [f'score_{label}_{int(alpha)}deg_{int(dist)}km' for label in ['shore', 'bedrock', 'wetland', 'beach'] for
               alpha, dist in zip(alphas, distances//1000)]

alphas *= np.pi / 180  # Convert to radians
n_alphas = len(alphas)

# Define the number of fields
num_fields = 4 + len(area_labels)

# Create a 2D numpy array
n = len(df_states)
new_features = np.zeros((n, num_fields), dtype=output_dtype)

# Fill the initial values
new_features[:, 0] = 0  # 'shore_rmsd'
new_features[:, 1] = init_distance  # 'shortest_distance'
new_features[:, 2] = init_distance  # 'shortest_distance_e'
new_features[:, 3] = init_distance  # 'shortest_distance_n'

# Set initial values for all the labels in area_labels
for i in range(4, num_fields):
    new_features[:, i] = 0


# %% Calculate the amount of shore that is upwind

@njit
def simple_linear_regression(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x * x)
    # calculate slope (b) and intercept (a)
    denominator = n * sum_xx - sum_x ** 2
    if denominator == 0:
        b = 0
    else:
        b = (n * sum_xy - sum_x * sum_y) / denominator
    a = (sum_y - b * sum_x) / n
    return a, b


@njit(parallel=True)
def add_shore_features(lat_state, lon_state, lat_shore, lon_shore, lat_cm, lon_cm, type_cm, u10, v10, new_features,
                       alphas, distances):
    """
    Add features to the new_features array
    :param lat_state:   # degrees
    :param lon_state:   # degrees
    :param lat_shore:   # degrees
    :param lon_shore:   # degrees
    :param lat_cm:   # radians
    :param lon_cm:   # radians
    :param type_cm:    # 0: 'Bedrock', 1: 'Wetland', 2: 'Beach'
    :param u10:    # m/s
    :param v10:    # m/s
    :param aprox_dists:   # km
    :param new_features:
    :return: filled array new_features
    """

    lat_state = np.radians(lat_state)
    lon_state = np.radians(lon_state)
    lat_shore = np.radians(lat_shore)
    lon_shore = np.radians(lon_shore)
    lat_cm = np.radians(lat_cm)
    lon_cm = np.radians(lon_cm)

    # Strangely, somehow sometimes the shore is not found in the box around the coordinate!
    no_near_shore_mask = np.zeros(len(lat_state), dtype=np.bool_)
    R_earth = 6371.0  # km
    side_length = np.sqrt(2) * 24  # km
    # Loop over all segments
    for i_state in prange(len(lat_state)):
        lon = lon_state[i_state]
        lat = lat_state[i_state]
        wind_u = u10[i_state]
        wind_v = v10[i_state]
        # Get shore points in a box around the coordinate
        lon_length = side_length / (R_earth * np.cos(lat))  # adjust for longitude
        min_lon = lon - lon_length
        if min_lon < -np.pi:
            min_lon += 2 * np.pi
        max_lon = lon + lon_length
        if max_lon > np.pi:
            max_lon -= 2 * np.pi

        lat_length = side_length / R_earth
        min_lat = lat - lat_length
        max_lat = lat + lat_length

        mask_lon = (lon_shore >= min_lon) & (lon_shore <= max_lon)
        mask_lat = (lat_shore >= min_lat) & (lat_shore <= max_lat)
        mask = mask_lon & mask_lat

        if np.sum(mask):
            lon_shore_state = lon_shore[mask]
            lat_shore_state = lat_shore[mask]

            # Add the 'straightness of the shore' as a feature
            a, b = simple_linear_regression(lon_shore_state, lat_shore_state)
            predicted = a + b * lon_shore_state
            new_features[i_state, 0] = np.sqrt(np.mean((lat_shore_state - predicted) ** 2))

            dlon = lon_shore_state - lon
            dlat = lat_shore_state - lat

            dxs = R_earth * np.cos(lat) * dlon
            dys = R_earth * dlat

            distances_shore_state = np.sqrt(dxs ** 2 + dys ** 2)

            # Save the shortest distance and direction to the shore
            # - distance:
            shortest_distance = np.min(distances_shore_state)
            new_features[i_state, 1] = shortest_distance  # 'shortest_distance'

            # - direction:
            i_nearest_shore_point = np.argmin(distances_shore_state)
            new_features[i_state, 2] = dxs[i_nearest_shore_point]
            new_features[i_state, 3] = dys[i_nearest_shore_point]

            # Calculate wind direction from point i to j
            wind_angle = math.atan2(wind_v, wind_u)
            angles = np.abs(np.arctan2(dys, dxs) - wind_angle)
            for angle, distance in zip(angles, distances_shore_state):
                for i_area, (alpha, radius) in enumerate(zip(alphas, distances)):
                    if distance <= radius and angle <= alpha / 2:
                        new_features[i_state, i_area] += 1 - distance / radius

            # Follow similar logic for coastal geomorphology:
            # Get a box around the coordinate
            mask_lon = (lon_cm >= min_lon) & (lon_cm <= max_lon)
            mask_lat = (lat_cm >= min_lat) & (lat_cm <= max_lat)
            mask = mask_lon & mask_lat
            lon_cm_box = lon_cm[mask]
            lat_cm_box = lat_cm[mask]
            type_cm_box = type_cm[mask]

            dlon = lon_cm_box - lon
            dlat = lat_cm_box - lat

            dxs = R_earth * np.cos(lat) * dlon
            dys = R_earth * dlat

            distances_cm_state = np.sqrt(dxs ** 2 + dys ** 2)
            angles = np.abs(np.arctan2(dys, dxs) - wind_angle)
            for angle, type_c, distance in zip(angles, type_cm_box, distances_cm_state):
                for i_area, (alpha, radius) in enumerate(zip(alphas, distances)):
                    if distance <= radius and angle <= alpha / 2:
                        i_label = i_area + (type_c + 1) * n_alphas
                        new_features[i_state, i_label] += 1 - distance / radius

        else:
            no_near_shore_mask[i_state] = True

    return new_features, no_near_shore_mask


features, no_near_shore_mask = add_shore_features(df_states.latitude.values, df_states.longitude.values,
                                                  df_shore.latitude.values, df_shore.longitude.values,
                                                  df_cm.latitude.values, df_cm.longitude.values, df_cm.type.values,
                                                  df_states.wind10m_u_mean.values, df_states.wind10m_v_mean.values,
                                                  new_features, alphas, distances*1000)

df_new = pd.DataFrame(data=features, columns=['shore_rmsd', 'shore_distance', 'shore_distance_e', 'shore_distance_n'] + area_labels)

df_state_new = pd.concat([df_states[~no_near_shore_mask].reset_index(drop=True),
                          df_new[~no_near_shore_mask].reset_index(drop=True)],
                         axis=1)

# %% Drop segments where no shore was found (for some reason!!!)
with open('deleted_segments_no_shore_found.txt', 'w') as f:
    f.write(f'Number of deleted segments because no shore was found: {sum(no_near_shore_mask)}\n'
            '-----------------------------------------------------\n')
    for i, b in enumerate(no_near_shore_mask):
        if b:
            f.write(f'{i}\n')

print('Number of deleted segments because no shore was found: ', sum(no_near_shore_mask))
print('Remaining number of beaching events:', df_state_new['beaching_flag'].sum())

# %% Calculate the inner-product of the wind vector and the vector to the nearest shore
df_state_new.to_csv(file_name_3, index=False)
print(f'Features added to {file_name_3}')

