import pandas as pd
from file_names import *


df1 = pd.read_csv(file_name_1, parse_dates=['time'], infer_datetime_format=True)
df1 = df1[df1.time >= '1993-01-01']

time_plus_1h = df1['time'] + pd.Timedelta('1h')
mask = time_plus_1h.dt.month == df1['time'].dt.month
df1 = df1[mask]

df2 = pd.read_csv(file_name_2, parse_dates=['time'], infer_datetime_format=True)

df2 = df2.merge(df1, how='left')

df2.to_csv(file_name_2, index=False)