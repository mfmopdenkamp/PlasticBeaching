import picklemanager as pickm
import load_data

#%%%
df_shoreline = load_data.get_shoreline('i')




#%%
d_gdf_shoreline = {}
for resolution in ['c', 'l', 'i', 'h', 'f']:
    d_gdf_shoreline[resolution] = pickm.pickle_wrapper('shoreline_' + resolution, load_data.get_shoreline, resolution)
    d_gdf_shoreline[resolution].to_crs(crs=3857, inplace=True)

for key in d_gdf_shoreline:
    print(f'{key} : length = {d_gdf_shoreline[key].geometry.exterior.length.sum()}')
