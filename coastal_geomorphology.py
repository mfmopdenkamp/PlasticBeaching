
import load_data as ld


gdf_cm = ld.get_coastal_morphology(points_only=True)

# read shapely file
print(gdf_cm.head())
print(gdf_cm.columns)
print(gdf_cm.crs)