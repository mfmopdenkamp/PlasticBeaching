import load_data
import matplotlib.pyplot as plt

df_shore = load_data.get_shoreline(resolution='l')

min_lon = 3
max_lon = 6
min_lat = 50
max_lat = 54

selected_gdf = df_shore[(df_shore.bounds['minx'] >= min_lon) & (df_shore.bounds['maxx'] <= max_lon) &
                  (df_shore.bounds['miny'] >= min_lat) & (df_shore.bounds['maxy'] <= max_lat)]

fig, ax = plt.subplots()
selected_gdf.plot(ax=ax)
plt.xlabel(r'longitude')
xticklabels = ax.get_xticklabels()
xticklabels = [label.get_text() + '°' for label in xticklabels]
ax.set_xticklabels(xticklabels)

plt.xlabel(r'latitude')
yticklabels = ax.get_yticklabels()
yticklabels = [label.get_text() + '°' for label in yticklabels]
ax.set_yticklabels(yticklabels)

plt.show()
