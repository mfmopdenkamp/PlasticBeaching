import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import load_data
print('0')

gdp = load_data.drifter_data_six_hourly(30000)

ax = plt.axes(projection=ccrs.PlateCarree())
for ID in gdp['ID'].unique():
    ax.plot(gdp[gdp['ID'] == ID]['Longitude'], gdp[gdp['ID'] == ID]['Latitude'], transform=ccrs.PlateCarree())
    break

ax.coastlines()
plt.show()


