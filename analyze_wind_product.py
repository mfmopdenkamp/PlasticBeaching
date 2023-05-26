import numpy as np
import pandas as pd
import math
import load_data
import analyzer as a

basic_file = 'segments_24h_subset_100_gps_undrogued_12km'
wind_file = basic_file + '_wind.csv'
wind2_file = basic_file + '_wind2.csv'

df = pd.read_csv('data/' + wind_file, parse_dates=['time_start', 'time_end'], index_col='ID')
df_shoreline = load_data.get_shoreline('f', points_only=True)

# %% Calculate the inproduct of the wind vector and the vector to the nearest shore
inproducts = np.empty(len(df))
for i, (de, dn, d, u, v) in enumerate(zip(df['de'], df['dn'], df['distance_nearest_shore'], df['u_mean'], df['v_mean'])):
    inproducts[i] = de/d*u + dn/d*v

df['inproduct_wind_nearest_shore'] = inproducts

# %% Calculate the amount of shore that is upwind
df_gdp = a.ds2geopandas_dataframe(df['latitude_start'], df['longitude_start'], df_shoreline)


def get_points_in_wind_direction(df_gdp, df_shore, alpha, side_length):

    x_drifter = df_gdp.geometry.x.values
    y_drifter = df_gdp.geometry.y.values

    count = np.zeros(len(x_drifter), dtype=np.float16)
    for i, (x_d, y_d, u, v, lon, lat) in enumerate(zip(x_drifter, y_drifter, df['u_mean'], df['v_mean'],
                                                  df['longitude_start'], df['latitude_start'])):

        # get shore points in a box around the coordinate
        min_lon, max_lon, min_lat, max_lat = a.get_lonlatbox(lon, lat, side_length*2)

        df_shore_box = df_shore[(df_shore['longitude'] >= min_lon) & (df_shore['longitude'] <= max_lon) &
                                (df_shore['latitude'] >= min_lat) & (df_shore['latitude'] <= max_lat)]

        x_shore = df_shore_box.geometry.x.values
        y_shore = df_shore_box.geometry.y.values

        for x_s, y_s in zip(x_shore, y_shore):
            # Calculate distance between points i and j
            dx = x_s - x_d
            dy = y_s - y_d
            distance = np.hypot(dx, dy)
            if distance <= side_length:
                # Calculate wind direction from point i to j
                wind_direction = math.atan2(v, u)

                # Calculate angle between wind direction and line connecting points i and j
                angle = abs(math.atan2(dy, dx) - wind_direction)
                if angle > math.pi:
                    angle = 2*math.pi - angle

                # Check if angle and distance are within the bounds of the isosceles triangle
                if angle <= alpha/2:
                    count[i] += 1-distance/side_length

    return count

alpha = 45
distance = 10000
degrees = 360/np.array([16, 8, 6, 4, 2, 4/3, 1])
var_distances = distance*np.sqrt(alpha/degrees)

for degree, var_dist in zip(degrees, var_distances):
    df[f'upwind_shore_counts_{degree}_{var_dist}'] = get_points_in_wind_direction(df_gdp, df_shoreline,
                                                                                  degree/180*math.pi, var_dist)
    df[f'upwind_shore_counts_{degree}_{distance}'] = get_points_in_wind_direction(df_gdp, df_shoreline,
                                                                                  degree / 180 * math.pi, distance)
df.to_csv('data/' + wind2_file)
