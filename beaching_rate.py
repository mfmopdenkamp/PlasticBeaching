import load_data
import numpy as np
import pickle_manager as pickm
import matplotlib.pyplot as plt
from shapely.geometry import Point
from tqdm import tqdm

# load hourly GDP of drifters 10km within the coast
ds = ds_subset = pickm.load_pickle('pickledumps/ds_gdp_subset_10km_distances.pkl')

# select only death type == 3
ds = ds.isel(traj=np.where(ds.type_death == 3)[0])

df_shore = load_data.get_shoreline('h')

ds_bath = load_data.get_bathymetry()

ids = np.unique(ds.ids)
n = len(ids)

beaching_criterion = {criterion: np.zeros(n, dtype=bool) for criterion in ['proximity', 'depth', 'kaandorp']}

# loop over every trajectory
for i, ID in enumerate(tqdm(ids)):
    ds_id = ds.isel(obs=np.where(ds.ids == ID)[0])
    lon_end = ds_id.longitude[-1]
    lat_end = ds_id.latitude[-1]

    # Test proximity criterion
    end_position = Point(lon_end, lat_end)
    prox_bool = df_shore.geometry.intersects(end_position)[0]
    if prox_bool:
        beaching_criterion['proximity'][i] = prox_bool
    else:
        print('\nNo proximity. Testing bathymetry..')
        # Test depth criterion
        end_depth = ds_bath.interp(coords={'lon': lon_end,
                                           'lat': lat_end},
                                   method='linear')['elevation'].values
        depth_bool = True if end_depth > -30 else False
        if depth_bool:
            beaching_criterion['depth'][i] = depth_bool
        else:
            print('Testing Kaandorp..')
            # Test Kaandorp criterion
            neighbouring_points = np.zeros((4,))
            displacement = 0.00833333  # 30 arc-seconds in degrees
            neighbouring_points[0] = ds_bath.interp(coords={'lon': lon_end - displacement, 'lat': lat_end}, method='linear')[
                'elevation'].values
            neighbouring_points[1] = ds_bath.interp(coords={'lon': lon_end + displacement, 'lat': lat_end}, method='linear')[
                'elevation'].values
            neighbouring_points[2] = ds_bath.interp(coords={'lon': lon_end, 'lat': lat_end - displacement}, method='linear')[
                'elevation'].values
            neighbouring_points[3] = ds_bath.interp(coords={'lon': lon_end, 'lat': lat_end + displacement}, method='linear')[
                'elevation'].values

            beaching_criterion['kaandorp'][i] = True if np.max(neighbouring_points) > 0 else False

