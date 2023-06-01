import numpy as np
import pandas as pd
import math
import load_data
import toolbox as tb

basic_file = 'segments_24h_subset_100_gps_undrogued_12km'
wind_file = basic_file + '_wind.csv'
wind2_file = basic_file + '_wind2.csv'

df_seg = pd.read_csv('data/' + wind_file, parse_dates=['time_start', 'time_end'], index_col='ID')
gdf_shore = load_data.get_shoreline('f', points_only=True)
gdf_cm = load_data.get_coastal_morphology(points_only=True)

# %% Calculate the inproduct of the wind vector and the vector to the nearest shore
inproducts = np.empty(len(df_seg))
for i, (de, dn, d, u, v) in enumerate(zip(df_seg['de'], df_seg['dn'], df_seg['distance_nearest_shore'], df_seg['wind10m_u_mean'],
                                          df_seg['wind10m_v_mean'])):
    inproducts[i] = de/d*u + dn/d*v

df_seg['inproduct_wind_nearest_shore'] = inproducts

# %% Calculate the amount of shore that is upwind
def get_points_in_wind_direction(gdf_seg, gdf_shore, alphas, side_lengths):

    x_drifter = gdf_seg.geometry.x.values
    y_drifter = gdf_seg.geometry.y.values

    n = len(x_drifter)
    output_dtype = np.float32
    scores = {}
    for alpha, side_length in zip(alphas, side_lengths):
        alpha_print = f'{alpha / math.pi / 2 * 360:.0f}deg'
        side_length_print = f'{side_length / 1000:.0f}km'
        shore_label = f'score_shore_{alpha_print}_{side_length_print}'
        bedrock_label = f'score_bedrock_{alpha_print}_{side_length_print}'
        wetland_label = f'score_wetland_{alpha_print}_{side_length_print}'
        beach_label = f'score_beach_{alpha_print}_{side_length_print}'

        scores[shore_label] = np.zeros(n, dtype=output_dtype)
        scores[bedrock_label] = np.zeros(n, dtype=output_dtype)
        scores[wetland_label] = np.zeros(n, dtype=output_dtype)
        scores[beach_label] = np.zeros(n, dtype=output_dtype)

    for i, (x_d, y_d, wind_u, wind_v, lon, lat) in enumerate(zip(x_drifter, y_drifter, df_seg['wind10m_u_mean'],
                                                                 df_seg['wind10m_v_mean'],
                                                                 gdf_seg['longitude'], gdf_seg['latitude'])):

        # get shore points in a box around the coordinate
        min_lon, max_lon, min_lat, max_lat = tb.get_lonlatbox(lon, lat, 24000)  # arbitrary 24km box

        gdf_shore_box = gdf_shore[(gdf_shore['longitude'] >= min_lon) & (gdf_shore['longitude'] <= max_lon) &
                                 (gdf_shore['latitude'] >= min_lat) & (gdf_shore['latitude'] <= max_lat)]

        gdf_cm_box = gdf_cm[(gdf_cm['longitude'] >= min_lon) & (gdf_cm['longitude'] <= max_lon) &
                                    (gdf_cm['latitude'] >= min_lat) & (gdf_cm['latitude'] <= max_lat)]

        x_shore = gdf_shore_box.geometry.x.values
        y_shore = gdf_shore_box.geometry.y.values

        x_cm = gdf_cm_box.geometry.x.values
        y_cm = gdf_cm_box.geometry.y.values
        type_cm = gdf_cm_box.Preds.values

        for alpha, side_length in zip(alphas, side_lengths):

            for x_s, y_s in zip(x_shore, y_shore):
                # Calculate distance between points i and j
                dx = x_s - x_d
                dy = y_s - y_d
                distance = np.hypot(dx, dy)
                if distance <= side_length:
                    # Calculate wind direction from point i to j
                    wind_direction = math.atan2(wind_v, wind_u)

                    # Calculate angle between wind direction and line connecting points i and j
                    angle = abs(math.atan2(dy, dx) - wind_direction)
                    if angle > math.pi:
                        angle = 2*math.pi - angle

                    # Check if angle and distance are within the bounds of the isosceles triangle
                    if angle <= alpha/2:
                        scores[shore_label][i] += 1-distance/side_length

            for x_c, y_c, type_c in zip(x_cm, y_cm, type_cm):

                dx = x_c - x_d
                dy = y_c - y_d
                distance = np.hypot(dx, dy)
                if distance <= side_length:
                    # Calculate wind direction from point i to j
                    wind_direction = math.atan2(wind_v, wind_u)

                    # Calculate angle between wind direction and line connecting points i and j
                    angle = abs(math.atan2(dy, dx) - wind_direction)
                    if angle > math.pi:
                        angle = 2*math.pi - angle

                    # Check if angle and distance are within the bounds of the isosceles triangle
                    if angle <= alpha/2:
                        if type_c == 'Bedrock':
                            scores[bedrock_label][i] += 1-distance/side_length
                        elif type_c == 'Wetland':
                            scores[wetland_label][i] += 1-distance/side_length
                        elif type_c == 'Beach':
                            scores[beach_label][i] += 1-distance/side_length
                        else:
                            raise ValueError(f'Unknown type {type_c}')
    return scores


gdf_segments = tb.dataset2geopandas(df_seg['latitude_start'], df_seg['longitude_start'], gdf_shore)

alpha = 45
distance = 10000
degrees = 360/np.array([16, 8, 6, 4, 2, 4/3, 1])
var_distances = distance*np.sqrt(alpha/degrees)

for degree, var_dist in zip(degrees, var_distances):
    df_seg[f'upwind_shore_counts_{degree}_{var_dist}'] = get_points_in_wind_direction(gdf_segments, gdf_shore,
                                                                                      degree / 180 * math.pi, var_dist)
    df_seg[f'upwind_shore_counts_{degree}_{distance}'] = get_points_in_wind_direction(gdf_segments, gdf_shore,
                                                                                      degree / 180 * math.pi, distance)
df_seg.to_csv('data/' + wind2_file, index_label='ID')
