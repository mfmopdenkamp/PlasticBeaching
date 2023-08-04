import picklemanager as pm
import pandas as pd
import numpy as np

gdf_shore = pm.load_pickle(pm.create_pickle_path('shoreline_f_points'))
df_shore_lonlat = pd.DataFrame({'longitude': gdf_shore.longitude.values, 'latitude': gdf_shore.latitude.values})

gdf_cm = pm.load_pickle(pm.create_pickle_path('coastal_morphology_points'))
map_coasts_to_int = {'Bedrock': 0, 'Wetland': 1, 'Beach': 2}
vfunc = np.vectorize(lambda x: map_coasts_to_int[x])
cm_type = vfunc(gdf_cm.Preds.values)
df_cm_lonlat = pd.DataFrame({'longitude': gdf_cm.longitude.values, 'latitude': gdf_cm.latitude.values,
                             'type': cm_type})


pm.dump_pickle(df_shore_lonlat, base_name='df_shoreline_f_lonlat')
pm.dump_pickle(df_cm_lonlat, base_name='df_coastal_morphology_lonlat')

