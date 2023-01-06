import geopandas as gpd
import xarray as xr
import load_data

ds = load_data.drifter_data_hourly(filename='gdp_galapagos.nc')

gshhg = load_data.coast_lines()

# min_lat = -92
# max_lat = -76
# min_lon = -5
# max_lon = 10
# lonlat_box = {'longitude': (min_lon, max_lon), 'latitude': (min_lat, max_lat)}
# ds_subset = ds.sel(lonlat_box)
#
# mask = (ds.longitude >= min_lon) & (ds.longitude <= max_lon) & (ds.latitude >= min_lat) & (ds.latitude <= max_lat)
# ds_subset = ds.where(mask)

gdf = gpd.GeoDataFrame(ds.to_dataframe(), geometry=gpd.points_from_xy(ds.longitude, ds.latitude))

# Create a buffer around the shoreline with a distance of 10km
shoreline_buffer = gshhg.geometry.buffer(10000)

# Find the locations in the xarray data that intersect with the buffered shoreline
mask = gdf.geometry.intersects(shoreline_buffer)

# Select only the locations within 10km of the shoreline from the original xarray data
selected_data = ds.where(mask)