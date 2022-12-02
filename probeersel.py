import pandas as pd
import os


def load_data(version='six hourly', nrows=None, parts=(1, 2, 3, 4), meta=False):

    with open(f'data/header_{("meta" if meta else "")}data.txt', 'rt') as f:
        data_header = f.readline().split(',')

    data_folder = 'data/'
    filenames = {1:'buoydata_1_5000.dat.gz',
                 2:'buoydata_5001_10000.dat.gz',
                 3:'buoydata_10001_15000.dat.gz',
                 4:'buoydata_15001_jul22.dat.gz'}

    df = pd.DataFrame()
    # kwargs = {'parse_dates': [[1, 2, 3]], 'names': data_header, 'delim_whitespace': True}
    kwargs = {'dtype':{'month': str, 'day': str, 'year':str}, 'names': data_header, 'delim_whitespace': True}
    if nrows:
        kwargs['nrows'] = nrows

    for part in parts:
        df = pd.concat((df, pd.read_csv(data_folder + filenames[part], **kwargs)))

    df[['day', 'part of day']] = df['day'].str.split('.', 1, expand=True)
    df['hour'] = df['part of day'].map(lambda x: str(int(x)*24//1000).zfill(2))
    df['day'] = df['day'].map(lambda x: x.zfill(2))
    df['month'] = df['month'].map(lambda x: x.zfill(2))

    df['datetime'] = df['year'] + '-' + df['month'] + '-' + df['day'] + ' ' + df['hour']
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H')

    df.drop(columns=['month', 'day', 'year', 'hour', 'part of day'], inplace=True)

    return df


df = load_data(nrows=10)

with open('data/buoydata_1_5000.dat', 'rt') as f:
    for _ in range(10):
        print(f.readline())

with open('data/buoydata_5001_10000.dat', 'rt') as f:
    print(f.readline())
