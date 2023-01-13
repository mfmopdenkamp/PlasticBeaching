import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import plotter
import load_data
import pickle_manager as pickm

filename = 'gdp_galapagos.nc'
ds_drifters = pickm.load_pickle_wrapper(filename, load_data.drifter_data_hourly, filename)

nrows=1000000
df_drifters = pickm.load_pickle_wrapper(f'drifter_data_{nrows}', load_data.drifter_data_six_hourly, nrows)

# min_lat = -92
# max_lat = -76
# min_lon = -5
# max_lon = 10
# lonlat_box = {'longitude': (min_lon, max_lon), 'latitude': (min_lat, max_lat)}
# ds_drifters = ds_drifters.sel(lonlat_box)

gdf_drifters = gpd.GeoDataFrame(ds_drifters.to_dataframe(),
                                geometry=gpd.points_from_xy(ds_drifters.longitude, ds_drifters.latitude),
                                crs='epsg:4326')

resolution = 'l'
gdf_shoreline = pickm.load_pickle_wrapper(f'shoreline_{resolution}', load_data.shoreline, resolution)

gdf_shoreline.to_crs(crs=3857, inplace=True)
gdf_drifters.to_crs(gdf_shoreline.crs, inplace=True)


nearby_drifters = gpd.tools.sjoin(gdf_drifters, gdf_shoreline, predicate='within')
distance = 0.1
mask = gdf_drifters.geometry.intersects(gdf_shoreline.buffer(0.1))
while mask.sum() <= 0:
    distance *= 10
    print(f'Mask is empty. New distance = {distance}')
    mask = gdf_drifters.geometry.intersects(gdf_shoreline.buffer(distance))
distance *= 10
print(f'New distance = {distance}')
mask = gdf_drifters.geometry.intersects(gdf_shoreline.buffer(distance))

distance2 = 10
mask2 = gdf_drifters.geometry.distance(gdf_shoreline.geometry) < distance2
while mask2.sum() <= 0:
    distance2 *= 10
    print(f'Mask is empty. New distance2 = {distance2}')
    mask2 = gdf_drifters.geometry.distance(gdf_shoreline.geometry) < distance2
distance2 *= 10
print(f'New distance2 = {distance2}')
mask2 = gdf_drifters.geometry.distance(gdf_shoreline.geometry) < distance2


# fig = plt.figure(dpi=300)
# ax = plt.axes(projection=ccrs.PlateCarree())
fig, ax = plotter.get_sophie_subplots(extent=(-92.5, -88.5, -1.75, 1))
ax.scatter(gdf_drifters['longitude'], gdf_drifters['latitude'], transform=ccrs.PlateCarree(), c='k', s=1,
           label='all', alpha=0.3)
ax.scatter(gdf_drifters[mask]['longitude'], gdf_drifters[mask]['latitude'], transform=ccrs.PlateCarree(),
           label='intersect')
ax.scatter(gdf_drifters[mask2]['longitude'], gdf_drifters[mask2]['latitude'], transform=ccrs.PlateCarree(),
           label='distance')
ax.scatter(nearby_drifters['longitude'], nearby_drifters['latitude'], transform=ccrs.PlateCarree(), label='sjoin')

ax.legend(loc=3)
plt.show()


# #
# # mask = (ds.longitude >= min_lon) & (ds.longitude <= max_lon) & (ds.latitude >= min_lat) & (ds.latitude <= max_lat)
# # ds_subset = ds.where(mask)
#


#
# # Create a buffer around the shoreline with a distance of 10km
# shoreline_buffer = gshhg.geometry.buffer(10000)
#
# # Find the locations in the xarray data that intersect with the buffered shoreline
# mask = gdf.geometry.intersects(shoreline_buffer)
#
# # Select only the locations within 10km of the shoreline from the original xarray data
# selected_data = ds.where(mask)