import load_data
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as crs

ds = load_data.drifter_data_hourly()
meta = load_data.drifter_metadata()
n_ids = len(meta)

death_types = np.sort(meta['Death Type'].unique())
death_type_fraction = np.zeros(len(death_types))
for d_type in death_types:
    death_type_fraction[d_type] = len(meta[meta['Death Type'] == d_type]) / n_ids

death_colors = {0: 'r', 1: 'g', 2: 'b', 3: 'y', 4: 'c', 5: 'm', 6: 'aquamarine'}
fig, ax = plt.subplots()
ax.bar(death_types, death_type_fraction, color=death_colors.values())
ax.set_ylabel('Fraction')
ax.set_xlabel('Death Type')
plt.show()

fig2 = plt.figure(figsize=(14, 8))
ax = plt.axes(projection=crs.PlateCarree())
for i_dt, d_type in enumerate(death_types):
    df_to_plot = meta[meta['Death Type'] == d_type]
    ax.scatter(df_to_plot['End Longitude'], df_to_plot['End Latitude'], color=death_colors[i_dt], s=2,
               transform=crs.PlateCarree(), label='d_type')

ax.coastlines()
ax.legend()
plt.show()
