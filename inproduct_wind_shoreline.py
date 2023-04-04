import numpy as np
import pandas as pd
import load_data

df = pd.read_csv('data/events_prep_non_splitted_drogued.csv', parse_dates=)

df_shoreline = load_data.get_shoreline('f', points_only=True)


for lat, lon in zip(df['latitude_start'], df['latitude_end']):
    pass