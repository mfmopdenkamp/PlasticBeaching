import numpy as np
import pandas as pd
import math
import config
import load_data
import analyzer as a

df = pd.read_csv('data/' + config.filename_wind, parse_dates=['time_start', 'time_end'])
df_shoreline = load_data.get_shoreline(config.shoreline_resolution, points_only=True)

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
        min_lon = lon - 0.15
        if min_lon < -180:
            min_lon += 360
        max_lon = lon + 0.15
        if max_lon > 180:
            max_lon -= 360
        min_lat = lat - 0.15
        max_lat = lat + 0.15

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


df['upwind_shore_counts'] = get_points_in_wind_direction(df_gdp, df_shoreline, math.pi/4, 10000)

df.to_csv('data/' + config.filename_wind + '_plus')
