import shapely as sh
import shapely.geometry as shg
import geopandas as gpd

file_name = 'data/CoastalGeomorphology/CoastalGeomorphology.shp'

gdf = gpd.read_file(file_name)

# read shapely file
