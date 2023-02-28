import pandas as pd
import geopandas as gpd
import picklemanager as pickm
import xarray as xr
import time
import os


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
    kwargs = {'parse_dates': {'Deployment':[4, 5], 'End':[8, 9], 'Drogue Off': [12, 13]},
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


def get_shoreline(resolution):
    return pickm.pickle_wrapper(f'shoreline_{resolution}', gpd.read_file,
                              f'{data_dir_name}gshhg-shp-2.3.7/GSHHS_shp/{resolution}/GSHHS_{resolution}_L1.shp')


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
