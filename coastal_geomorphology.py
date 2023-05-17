import shapely as sh
import shapely.geometry as shg
import geopandas as gpd

file_name = 'data/631485339c5c1bab_ECVGS2019_Q2903/631485339c5c1bab_ECVGS2019_Q2903/data/CoastalGeomorphology/CoastalGeomorphology.shp'

gdf = gpd.read_file(file_name)

# read shapely file
