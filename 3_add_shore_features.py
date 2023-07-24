import numpy as np
from numba import jit
import pandas as pd
import math
import load_data
import toolbox as tb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from file_names import *


def get_score_label(base, alpha, side_length):
    alpha_print = f'{alpha / math.pi / 2 * 360:.0f}deg'
    side_length_print = f'{side_length / 1000:.0f}km'
    return f'score_{base}_{alpha_print}_{side_length_print}'


# %% Calculate the amount of shore that is upwind
def add_shore_features(df_seg, gdf_seg, gdf_shore, gdf_cm, alphas, side_lengths):
    x_drifter = gdf_seg.geometry.x.values
    y_drifter = gdf_seg.geometry.y.values

    # Initialize output
    n = len(x_drifter)
    output_dtype = np.float32
    init_distance = np.finfo(output_dtype).max
    new_features = {'shore_rmsd': np.zeros(n, dtype=output_dtype),
                    'shortest_distance': np.zeros(n, dtype=output_dtype) * init_distance,
                    'shortest_distance_e': np.zeros(n, dtype=output_dtype) * init_distance,
                    'shortest_distance_n': np.zeros(n, dtype=output_dtype) * init_distance}

    for alpha, side_length in zip(alphas, side_lengths):
        new_features[get_score_label('shore', alpha, side_length)] = np.zeros(n, dtype=output_dtype)
        new_features[get_score_label('bedrock', alpha, side_length)] = np.zeros(n, dtype=output_dtype)
        new_features[get_score_label('wetland', alpha, side_length)] = np.zeros(n, dtype=output_dtype)
        new_features[get_score_label('beach', alpha, side_length)] = np.zeros(n, dtype=output_dtype)

    # Strangely, somehow sometimes the shore is not found in the box around the coordinate!
    no_near_shore_indexes = []

    # Loop over all segments
    for i_seg, (x_d, y_d, wind_u, wind_v, lon, lat) in enumerate(zip(x_drifter, y_drifter, df_seg['wind10m_u_mean'],
                                                                 df_seg['wind10m_v_mean'],
                                                                 gdf_seg['longitude'], gdf_seg['latitude'])):

        # Get shore points in a box around the coordinate
        min_lon, max_lon, min_lat, max_lat = tb.get_lonlatbox(lon, lat, 24000)  # arbitrary 24km box

        gdf_shore_box = gdf_shore[(gdf_shore['longitude'] >= min_lon) & (gdf_shore['longitude'] <= max_lon) &
                                  (gdf_shore['latitude'] >= min_lat) & (gdf_shore['latitude'] <= max_lat)]

        if not gdf_shore_box.empty:
            x_shore = gdf_shore_box.geometry.x.values
            y_shore = gdf_shore_box.geometry.y.values

            # fit linear shoreline direction
            lr = LinearRegression()
            lr.fit(x_shore.reshape(-1, 1), y_shore)
            new_features['shore_rmsd'][i_seg] = np.sqrt(mean_squared_error(y_shore, lr.predict(x_shore.reshape(-1, 1))))

            distances_shore_box = np.zeros(len(gdf_shore_box), dtype=output_dtype)
            for i_shore, (x_s, y_s) in enumerate(zip(x_shore, y_shore)):
                # Calculate distance between points i and j
                dx = x_s - x_d
                dy = y_s - y_d
                distance = np.hypot(dx, dy)
                distances_shore_box[i_shore] = distance

                for alpha, side_length in zip(alphas, side_lengths):
                    if distance <= side_length:
                        # Calculate wind direction from point i to j
                        wind_direction = math.atan2(wind_v, wind_u)

                        # Calculate angle between wind direction and line connecting points i and j
                        angle = abs(math.atan2(dy, dx) - wind_direction)
                        if angle > math.pi:
                            angle = 2 * math.pi - angle

                        # Check if angle and distance are within the bounds of the isosceles triangle
                        if angle <= alpha / 2:
                            new_features[get_score_label('shore', alpha, side_length)][i_seg] += 1 - distance / side_length

            # Save the shortest distance and direction to the shore
            # - distance:
            new_features['shortest_distance'][i_seg] = np.min(distances_shore_box)

            # - direction:
            i_nearest_shore_point = np.argmin(distances_shore_box)
            new_features['shortest_distance_n'][i_seg] = y_d - y_shore[i_nearest_shore_point]
            new_features['shortest_distance_e'][i_seg] = x_d - x_shore[i_nearest_shore_point]

            # Follow similar logic for coastal geomorphology:
            # Get a box around the coordinate
            gdf_cm_box = gdf_cm[(gdf_cm['longitude'] >= min_lon) & (gdf_cm['longitude'] <= max_lon) &
                                (gdf_cm['latitude'] >= min_lat) & (gdf_cm['latitude'] <= max_lat)]

            x_cm = gdf_cm_box.geometry.x.values
            y_cm = gdf_cm_box.geometry.y.values
            type_cm = gdf_cm_box.Preds.values

            for x_c, y_c, type_c in zip(x_cm, y_cm, type_cm):

                dx = x_c - x_d
                dy = y_c - y_d
                distance = np.hypot(dx, dy)

                for alpha, side_length in zip(alphas, side_lengths):

                    if distance <= side_length:
                        # Calculate wind direction from point i to j
                        wind_direction = math.atan2(wind_v, wind_u)

                        # Calculate angle between wind direction and line connecting points i and j
                        angle = abs(math.atan2(dy, dx) - wind_direction)
                        if angle > math.pi:
                            angle = 2 * math.pi - angle

                        # Check if angle and distance are within the bounds of the isosceles triangle
                        if angle <= alpha / 2:
                            if type_c == 'Bedrock':
                                new_features[get_score_label('bedrock', alpha, side_length)][i_seg] += 1 - distance / side_length
                            elif type_c == 'Wetland':
                                new_features[get_score_label('wetland', alpha, side_length)][i_seg] += 1 - distance / side_length
                            elif type_c == 'Beach':
                                new_features[get_score_label('beach', alpha, side_length)][i_seg] += 1 - distance / side_length
                            else:
                                raise ValueError(f'Unknown type {type_c}')
        else:
            no_near_shore_indexes.append(i_seg)

    if no_near_shore_indexes:
        print(f'No near shore found for segments : {no_near_shore_indexes}')

    return new_features, no_near_shore_indexes


df_seg2 = pd.read_csv(f'data/{file_name_2}.csv', parse_dates=['time_start', 'time_end'])
gdf_shore = load_data.get_shoreline('f', points_only=True)
gdf_cm = load_data.get_coastal_morphology(points_only=True)
gdf_segments = tb.dataset2geopandas(df_seg2['latitude_start'], df_seg2['longitude_start'], gdf_shore)

base_alpha = 45
base_distance = 10000
alphas = 360 / np.array([16, 8, 6, 4, 2, 4 / 3])
distances = base_distance * np.sqrt(base_alpha / alphas)

# Add full circles
alphas = np.append(alphas, np.array([360, 360, 360, 360]))
alphas *= np.pi / 180  # Convert to radians
distances = np.append(distances, np.ones(4) * base_distance * [1, 3/4, 1/2, 1/4])

features, no_near_shore_indexes = add_shore_features(df_seg2, gdf_segments, gdf_shore, gdf_cm, alphas, distances)
df_seg3 = pd.concat([df_seg2, pd.DataFrame.from_dict(features)], axis=1)

# %% Drop segments where no shore was found (for some reason!!!)
df_seg3.drop(no_near_shore_indexes, inplace=True)

print('Number of deleted segments because no shore was found: ', len(no_near_shore_indexes))
print('Remaining number of beaching events:', df_seg3['beaching_flag'].sum())


# %% Calculate the inner-product of the wind vector and the vector to the nearest shore
@jit(nopython=True)
def get_inproducts(x_shore, y_shore, wind_u, wind_v):
    inproducts = np.empty(len(x_shore))
    for i, (x, y, u, v) in enumerate(zip(x_shore, y_shore, wind_u, wind_v)):
        d = x**2 + y**2
        w = u**2 + v**2

        inproducts[i] = x / d * u / w + y / d * v / w
    return inproducts


# inproducts = np.empty(len(df_seg3))
# for i, (de, dn, d, u, v) in enumerate(
#         zip(df_seg3['shortest_distance_e'], df_seg3['shortest_distance_n'], df_seg3['shortest_distance'], df_seg3['wind10m_u_mean'],
#             df_seg3['wind10m_v_mean'])):
#     inproducts[i] = de / d * u + dn / d * v

df_seg3['inproduct_wind_nearest_shore'] = get_inproducts(df_seg3['shortest_distance_e'], df_seg3['shortest_distance_n'],
                                                         df_seg3['wind10m_u_mean'],
                                                         df_seg3['wind10m_v_mean'])


df_seg3.to_csv(f'data/{file_name_3}.csv', index_label='ID')
print(f'Features added to {file_name_3}.csv')

# print the numpy dtypes of the arrays in the dict
for key, value in features.items():
    print(key, value.dtype, value.shape[0])