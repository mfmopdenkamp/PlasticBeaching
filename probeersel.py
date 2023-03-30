
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV

df_prep = pd.read_csv('data/events_prep_non_splitted_drogued.csv').drop(columns='ID')
df_wind = pd.read_csv('data/events_wind.csv').drop(columns=['ID', 'Unnamed: 0'])
df = pd.merge(df_prep, df_wind)


cor = df.corr(numeric_only=True)

diki = cor['beaching_flags'].sort_values(ascending=False)[1:]


plt.bar(diki.index, diki)
plt.xticks(rotation=30)
plt.title('Beaching flag - Pearson correlation coefficients')
plt.tight_layout()
plt.show()

estimator = RandomForestClassifier()

