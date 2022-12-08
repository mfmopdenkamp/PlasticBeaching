import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def load_drifter_data(version='six hourly', nrows=None, parts=(1, 2, 3, 4)):
    with open(f'data/header_data.txt', 'rt') as f:
        data_header = f.readline().split(',')

    data_folder = 'data/'
    filenames = {1: 'buoydata_1_5000.dat.gz',
                 2: 'buoydata_5001_10000.dat.gz',
                 3: 'buoydata_10001_15000.dat.gz',
                 4: 'buoydata_15001_jul22.dat.gz'}
    kwargs = {'dtype': {'month': str, 'day': str, 'year': str}, 'names': data_header, 'delim_whitespace': True}
    if nrows:
        kwargs['nrows'] = nrows

    df = pd.DataFrame()
    for part in parts:
        df = pd.concat((df, pd.read_csv(data_folder + filenames[part], **kwargs)))

    df[['day', 'part of day']] = df['day'].str.split('.', expand=True)
    df['hour'] = df['part of day'].map(lambda x: str(int(x) * 24 // 1000).zfill(2))
    df['day'] = df['day'].map(lambda x: x.zfill(2))
    df['month'] = df['month'].map(lambda x: x.zfill(2))

    df['datetime'] = df['year'] + '-' + df['month'] + '-' + df['day'] + ' ' + df['hour']
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H')

    df.drop(columns=['month', 'day', 'year', 'hour', 'part of day'], inplace=True)
    df.set_index('datetime', inplace=True)

    return df


def load_drifter_metadata(nrows=None, parts=(1, 2, 3, 4)):
    with open(f'data/header_metadata.txt', 'rt') as f:
        data_header = f.readline().split(',')

    data_folder = 'data/'
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
        df = pd.concat((df, pd.read_csv(data_folder + filenames[part], **kwargs)))

    df['Drogue Off'] = pd.to_datetime(df['Drogue Off'], errors='coerce')

    return df


if __name__ == '__main__':
    df_data = load_drifter_data(nrows=3)
    df_meta = load_drifter_metadata()
