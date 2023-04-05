import pandas as pd
import geopandas as gpd
import picklemanager as pickm
import xarray as xr
import time
import os
import analyzer as a
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


def drifter_data_hourly(load_into_memory=True, filename='gdp_v2.00.nc'):
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


def _polygons_2_points(resolution):
    df = get_shoreline(resolution)

    shoreline_points = []
    lats = []
    lons = []

    for polygon in df.geometry:
        for coord in polygon.exterior.coords:
            shoreline_points.append(Point(coord))
            lons.append(coord[0])
            lats.append(coord[1])

    df = gpd.GeoDataFrame({'latitude': lats, 'longitude': lons}, geometry=shoreline_points, crs="WGS 84")

    # Use projected CRS ESPG:3857 for better distance calculations
    df.to_crs(crs=3857, inplace=True)
    return df


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
        df_shore = pickm.pickle_wrapper(f'shoreline_{resolution}_points', _polygons_2_points, resolution)
    else:
        df_shore = pickm.pickle_wrapper(f'shoreline_{resolution}', gpd.read_file,
                                        f'{data_dir_name}gshhg-shp-2.3.7/GSHHS_shp/{resolution}/GSHHS_{resolution}_L1'
                                        f'.shp')

    return df_shore


def get_bathymetry():
    filename_gebco = 'GEBCO_2022_sub_ice_topo.nc'
    return pickm.pickle_wrapper(filename_gebco, xr.load_dataset,
                                f'{data_dir_name}gebco_2022_sub_ice_topo/{filename_gebco}')


def get_ds_drifters(filename='gdp_v2.00.nc', proximity_of_coast=None, with_distances=False):
    if with_distances:
        return pickm.load_pickle(f'pickledumps/ds_gdp_subset_{proximity_of_coast}km_distances.pkl')
    if proximity_of_coast is not None:
        ds_subset = pickm.load_pickle(f'pickledumps/{filename}subset_{proximity_of_coast}km.pkl')
        return ds_subset
    return pickm.pickle_wrapper(filename, drifter_data_hourly, filename)


def load_random_subset(percentage=1):
    ds = get_ds_drifters('gdp_v2.00.nc_no_sst')

    n = len(ds.traj)
    traj = np.random.choice(np.arange(len(ds.traj)), size=int(n * percentage / 100), replace=False)
    obs = a.obs_from_traj(ds, traj)
    ds_subset = ds.isel(traj=traj, obs=obs)
    return ds_subset


def get_subtrajs(file_name='events_prep.csv'):
    df_prep = pd.read_csv('data/' + file_name, parse_dates=['time_start', 'time_end']).drop(
        columns='ID')
    df_wind = pd.read_csv('data/events_wind.csv', parse_dates=['time_start', 'time_end']).drop(
        columns=['ID', 'Unnamed: 0'])
    df = pd.merge(df_prep, df_wind)

    return df


if __name__ == '__main__':
    # print('Loading six-hourly drifter data into Pandas DataFrame..', end='')
    # start = time.time()
    # df_drifter_six_hourly = drifter_data_six_hourly()
    # print(f'Done. Elapsed time = {time.time()-start}')

    print('Loading hourly drifter data into Xarray..', end='')
    start = time.time()
    ds = drifter_data_hourly()
    print(f'Done. Elapsed time = {time.time() - start}')

    # print('Load drifter metadata..', end='')
    # df_meta = drifter_metadata()
    # print('Done.')

    # print('Load coastlines..', end='')
    # sf = coast_lines()
    # print('Done.')
