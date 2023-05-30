import shapely as sh
import shapely.geometry as shg
import geopandas as gpd
import picklemanager as pm

file_name = 'data/631485339c5c1bab_ECVGS2019_Q2903/631485339c5c1bab_ECVGS2019_Q2903/data/CoastalGeomorphology/CoastalGeomorphology.shp'
pickle_name = 'coastal_geomorphology'
gdf = pm.pickle_wrapper(pickle_name, gpd.read_file, file_name)

# read shapely file

