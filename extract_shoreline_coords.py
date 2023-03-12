import load_data
from shapely.geometry import Point
import geopandas as gpd

df_shore = load_data.get_shoreline(resolution='c')

# df_shore.to_crs(crs=3857, inplace=True)

shoreline_points = []
latitude = []
longitude = []
for polygon in df_shore.geometry:
    for coord in polygon.exterior.coords:
        shoreline_points.append(Point(coord))
        longitude.append(coord[0])
        latitude.append(coord[1])

df = gpd.GeoDataFrame(geometry=shoreline_points, crs="EPSG:4326")
