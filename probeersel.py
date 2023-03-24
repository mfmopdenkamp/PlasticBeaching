
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_prep = pd.read_csv('data/events_prep.csv').drop(columns='ID')
df_wind = pd.read_csv('data/events_wind.csv').drop(columns=['ID', 'Unnamed: 0'])
df = pd.merge(df_prep, df_wind)


cor = df.corr(numeric_only=True)

diki = cor['beaching_flags'].sort_values(ascending=False)[1:]


plt.bar(diki.index, diki)
plt.xticks(rotation=20)
plt.show()