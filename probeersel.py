
import pandas as pd
import numpy as np

df = pd.read_csv('data/events.csv', parse_dates=['time_start', 'time_end'], infer_datetime_format=True)

kaasie = np.zeros(df.shape[0])
for event in df.itertuples():
    print(f'Do something from start time : {event.time_start} until : {event.time_end}')
    months = pd.date_range(event.time_start, event.time_end + pd.DateOffset(months=1), freq='M')
    print(months)