import pandas as pd
import geopandas as gpd
import picklemanager as pickm
import xarray as xr
import time
import os
import toolbox as tb
import numpy as np
from shapely.geometry import Point


def find_data_directory():
    data_dir_name = 'data/'
    new_dir_name = data_dir_name
    max_levels = 3
    level = 0
    while not os.path.exists(new_dir_name) and level <= max_levels:
        level += 1
        new_dir_name = ''.join(['../', new_dir_name])

    if level > max_levels:
        print('No data folder found. Creating new one?')
        answer = ''
        while answer not in ['y', 'n']:
            answer = input('[y]/[n]?')
        if answer == 'y':
            os.mkdir(data_dir_name)
    else:
        data_dir_name = new_dir_name

    return data_dir_name


data_dir_name = find_data_directory()


def drifter_data_hourly(load_into_memory=False, filename='gdp_v2.00.nc'):
    f = data_dir_name + filename
    if load_into_memory:
        ds = xr.load_dataset(f, decode_cf=True, decode_times=False)
    else:
        ds = xr.open_dataset(f)
    return ds


def drifter_data_six_hourly(nrows=None, parts=(1, 2, 3, 4)):
    """
    Load the six-hourly Global Drifter Program data into a dataframe
    :param nrows: how many rows to read from each part.
    Note that giving nrows = x will result in len(df) = x * number of parts
    :param parts: The GDP data is divided into 4 subsets (parts).
    :return: Pandas DataFrame
    """

    directory = data_dir_name + 'gdp_six_hourly/'
    with open(f'{directory}header_data.txt', 'rt') as f:
        data_header = f.readline().split(',')

    filenames = {1: 'buoydata_1_5000.dat.gz',
                 2: 'buoydata_5001_10000.dat.gz',
                 3: 'buoydata_10001_15000.dat.gz',
                 4: 'buoydata_15001_jul22.dat.gz'}
    kwargs = {'dtype': {'month': str, 'day': str, 'year': str}, 'names': data_header, 'delim_whitespace': True}
    if nrows:
        kwargs['nrows'] = nrows

    df = pd.DataFrame()
    for part in parts:
        df = pd.concat((df, pd.read_csv(directory + filenames[part], **kwargs)))

    df[['day', 'part of day']] = df['day'].str.split('.', expand=True)
    df['hour'] = df['part of day'].map(lambda x: str(int(x) * 24 // 1000).zfill(2))
    df['day'] = df['day'].map(lambda x: x.zfill(2))
    df['month'] = df['month'].map(lambda x: x.zfill(2))

    df['datetime'] = df['year'] + '-' + df['month'] + '-' + df['day'] + ' ' + df['hour']
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H')

    df.drop(columns=['month', 'day', 'year', 'hour', 'part of day'], inplace=True)
    df.set_index('datetime', inplace=True)

    return df


def drifter_metadata(nrows=None, parts=(1, 2, 3, 4)):
    directory = data_dir_name + 'gdp_metadata/'
    with open(f'{directory}header_metadata.txt', 'rt') as f:
        data_header = f.readline().split(',')

    filenames = {1: 'dirfl_1_5000.dat',
                 2: 'dirfl_5001_10000.dat',
                 3: 'dirfl_10001_15000.dat',
                 4: 'dirfl_15001_jul22.dat'}
    kwargs = {'parse_dates': {'Deployment': [4, 5], 'End': [8, 9], 'Drogue Off': [12, 13]},
              'names': data_header, 'delim_whitespace': True}
    if nrows:
        kwargs['nrows'] = nrows

    df = pd.DataFrame()
    for part in parts:
        df = pd.concat((df, pd.read_csv(directory + filenames[part], **kwargs)))

    df['Drogue Off'] = pd.to_datetime(df['Drogue Off'], errors='coerce')

    return df


def get_raster_distance_to_shore_04deg():
    filename_dist2shore = 'dist2coast.txt.bz2'
    return pickm.pickle_wrapper(filename_dist2shore, pd.read_csv, data_dir_name + filename_dist2shore,
                                delim_whitespace=True, names=['longitude', 'latitude', 'distance'],
                                header=None, compression='bz2')


def geometry2points(gdf):

    points = []
    data = {}
    for col in gdf.columns:
        data[col] = []
    data['latitude'] = []
    data['longitude'] = []

    for i, geo in enumerate(gdf.geometry):
        for coord in (geo.exterior.coords if geo.geom_type == 'Polygon' else geo.coords):
            points.append(Point(coord))
            data['longitude'].append(coord[0])
            data['latitude'].append(coord[1])
            for col in gdf.columns:
                data[col].append(gdf[col][i])

    gdf = gpd.GeoDataFrame(data, geometry=points, crs=gdf.crs)

    # Use projected CRS ESPG:3857 for better distance calculations
    gdf.to_crs(crs=3857, inplace=True)
    return gdf


def get_shoreline(resolution, points_only=False):
    """
    :param resolution: from high to low the options are:
        f : Full resolution.  These contain the maximum resolution
            of this data and has not been decimated.
        h : High resolution.  The Douglas-Peucker line reduction was
            used to reduce data size by ~80% relative to full.
        i : Intermediate resolution.  The Douglas-Peucker line reduction was
            used to reduce data size by ~80% relative to high.
        l : Low resolution.  The Douglas-Peucker line reduction was
            used to reduce data size by ~80% relative to intermediate.
        c : Crude resolution.  The Douglas-Peucker line reduction was
            used to reduce data size by ~80% relative to low.
    :param points_only:
    :return:
    """
    if points_only:
        pickle_name = f'shoreline_{resolution}_points'
        try:
            gdf = None
            gdf_shore = pickm.pickle_wrapper(f'shoreline_{resolution}_points', geometry2points, gdf)
        except:
            gdf = get_shoreline(resolution)
            gdf_shore = pickm.pickle_wrapper(f'shoreline_{resolution}_points', geometry2points, gdf)
    else:
        gdf_shore = pickm.pickle_wrapper(f'shoreline_{resolution}', gpd.read_file,
                                        f'{data_dir_name}gshhg-shp-2.3.7/GSHHS_shp/{resolution}/GSHHS_{resolution}_L1'
                                        f'.shp')

    return gdf_shore


def get_coastal_morphology(points_only=False):

    if points_only:
        pickle_name = 'coastal_morphology_points'
        try:
            gdf_cm = None
            gdf_cm = pickm.pickle_wrapper('coastal_morphology_points', geometry2points, gdf_cm)
        except:
            gdf_cm = get_coastal_morphology()
            gdf_cm = pickm.pickle_wrapper('coastal_morphology_points', geometry2points, gdf_cm)
    else:
        gdf_cm = pickm.pickle_wrapper('coastal_morphology', gpd.read_file,
                                      f'{data_dir_name}631485339c5c1bab_ECVGS2019_Q2903/631485339c5c1bab_ECVGS2019_Q2903/data/CoastalGeomorphology/CoastalGeomorphology.shp')

    return gdf_cm


def get_bathymetry():
    filename_gebco = 'GEBCO_2022_sub_ice_topo.nc'
    return pickm.pickle_wrapper(filename_gebco, xr.load_dataset,
                                f'{data_dir_name}gebco_2022_sub_ice_topo/{filename_gebco}')


def get_ds_drifters(filename='gdp_v2.00.nc_no_sst', proximity_of_coast=None, with_distances=False):
    if with_distances:
        return pickm.load_pickle(f'pickledumps/ds_gdp_subset_{proximity_of_coast}km_distances.pkl')
    if proximity_of_coast is not None:
        ds_subset = pickm.load_pickle(f'pickledumps/{filename}subset_{proximity_of_coast}km.pkl')
        return ds_subset
    return pickm.pickle_wrapper(filename, drifter_data_hourly, filename)


def load_subset(traj_percentage=100, location_type=None, drogued=None, max_aprox_distance_km=None, start_date=None,
                end_date=None, ds=None, min_aprox_distance_km=None, type_death=None):

        if ds is None:
            ds = get_ds_drifters('gdp_v2.00.nc_no_sst')

        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            obs_start = ds.obs[ds.time >= start_date]
            traj_start = tb.traj_from_obs(ds, obs_start)
            ds = ds.isel(obs=obs_start, traj=traj_start)

        if end_date is not None:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            obs_end = ds.obs[ds.time <= end_date]
            traj_end = tb.traj_from_obs(ds, obs_end)
            ds = ds.isel(obs=obs_end, traj=traj_end)

        if traj_percentage < 100:
            n = len(ds.traj)
            size = int(n * traj_percentage / 100)
            traj_random = np.random.choice(np.arange(len(ds.traj)), size=size, replace=False)
            obs_random = tb.obs_from_traj(ds, traj_random)
            ds = ds.isel(traj=traj_random, obs=obs_random)

        if location_type is not None:
            if isinstance(location_type, str):
                if location_type[0] in ['A', 'a']:  # Argos
                    location_type = False
                elif location_type[0] in ['G', 'g']:  # GPS
                    location_type = True
            traj_gps = np.where(ds.location_type.values == location_type)[0]
            obs_gps = tb.obs_from_traj(ds, traj_gps)
            ds = ds.isel(traj=traj_gps, obs=obs_gps)

        if type_death is not None:
            traj_death = ds.traj[ds.type_death.values == type_death]
            obs_death = tb.obs_from_traj(ds, traj_death)
            ds = ds.isel(traj=traj_death, obs=obs_death)

        if drogued is not None:
            if drogued:
                obs_undrogued = ds.obs[tb.get_drogue_presence(ds)]
            else:
                obs_undrogued = ds.obs[~tb.get_drogue_presence(ds)]
            traj_undrogued = tb.traj_from_obs(ds, obs_undrogued)
            ds = ds.isel(obs=obs_undrogued, traj=traj_undrogued)

        if max_aprox_distance_km is not None:
            obs_close2shore = ds.obs[ds.aprox_distance_shoreline.values < max_aprox_distance_km]
            traj_close2shore = tb.traj_from_obs(ds, obs_close2shore)
            ds = ds.isel(traj=traj_close2shore, obs=obs_close2shore)

        if min_aprox_distance_km is not None:
            obs_far2shore = ds.obs[ds.aprox_distance_shoreline.values > min_aprox_distance_km]
            traj_far2shore = tb.traj_from_obs(ds, obs_far2shore)
            ds = ds.isel(traj=traj_far2shore, obs=obs_far2shore)

        return ds


if __name__ == '__main__':
    # print('Loading six-hourly drifter data into Pandas DataFrame..', end='')
    # start = time.time()
    # df_drifter_six_hourly = drifter_data_six_hourly()
    # print(f'Done. Elapsed time = {time.time()-start}')

    print('Loading hourly drifter data into Xarray..', end='')
    start = time.time()
    ds_kaas = drifter_data_hourly()
    print(f'Done. Elapsed time = {time.time() - start}')

    # print('Load drifter metadata..', end='')
    # df_meta = drifter_metadata()
    # print('Done.')

    # print('Load coastlines..', end='')
    # sf = coast_lines()
    # print('Done.')
