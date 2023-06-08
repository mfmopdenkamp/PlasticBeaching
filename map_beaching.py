import pandas as pd
from file_names import *
import folium
from folium.plugins import HeatMap

df = pd.read_csv(f'data/{file_name_4}.csv')

num_boxes = 10
lat_box_size = 0.5
lon_box_size = 0.5

# Calculate the box indices for each coordinate
df['lat_box'] = ((df['latitude_start']) // lat_box_size)
df['lon_box'] = ((df['longitude_start']) // lon_box_size)

# Group the coordinates by box indices and count the number of coordinates in each box
grouped = df.groupby(['lat_box', 'lon_box']).size().reset_index(name='count')

# Create a folium map centered at the mean coordinates
center_lat = 61
center_lon = -21
map_obj = folium.Map(location=[center_lat, center_lon], zoom_start=5)

# Convert the grouped data into a list of coordinates and counts
heat_data = [[row['lat_box'] * lat_box_size + lat_box_size / 2,
              row['lon_box'] * lon_box_size + lon_box_size / 2,
              row['count']] for _, row in grouped.iterrows()]

# Add the heatmap layer to the map
HeatMap(heat_data).add_to(map_obj)

# Create a colorbar
min_count = grouped['count'].min()
max_count = grouped['count'].max()
colorbar = folium.LinearColormap(['blue', 'yellow', 'red'], vmin=min_count, vmax=max_count, caption='Count')
colorbar.add_to(map_obj)

# Add an interactive tooltip for each location
for _, row in df.iterrows():
    flag, lat, lon, drifter_id, time = row['beaching_flag'], row['latitude_start'], row['longitude_start'], row['drifter_id'], row['time_start']
    flag_color = 'red' if flag else 'blue'
    folium.Marker(location=[lat, lon], tooltip=f'ID:{drifter_id}@({lat}, {lon})_{time}',
                  icon=folium.Icon(icon='fa-duotone fa-buoy-mooring fa-2xs', prefix='fa', color=flag_color)).add_to(map_obj)

# Display the map
map_obj.save('beaching_map.html')


