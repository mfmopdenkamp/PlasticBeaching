import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="iteritems")
import load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import scipy.stats as stats
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV

df = load_data.get_subtrajs(file_name='events_prep_non_splitted_drogued.csv')

cor = df.corr(numeric_only=True)

y_column = 'beaching_flags'

diki = pd.DataFrame({'pd.corr' : cor[y_column].sort_values(ascending=False)[1:]})


plt.bar(diki.index, diki['pd.corr'])
plt.xticks(rotation=30)
plt.title('Beaching flag - correlation coefficients')
plt.tight_layout()
plt.show()

n = diki.shape[0]
S = np.zeros(n)
S_p = np.zeros(n)

for i, column in enumerate(diki.index):
    S[i], S_p[i] = stats.pointbiserialr(df[y_column], df[column])

diki['pointbiserial'] = S
diki['pointbiserial_p'] = S_p

x = df.drop(columns=[y_column, 'time_start', 'time_end'])
y = df[y_column]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42, shuffle=False)

param_grid = {'n_estimators':[10, 50, 100], 'min_samples_split':[2, 5, 10, 20]}
estimator = RandomForestClassifier()
grid_search = HalvingGridSearchCV(estimator, param_grid=param_grid, verbose=2)
grid_search.fit(x, y)
df_gs = pd.DataFrame(grid_search.cv_results_)

estimator.fit(x_train, y_train)

y_pred = estimator.predict(x_test)

a_score = accuracy_score(y_test, y_pred)

c_matrix = confusion_matrix(y_test, y_pred)
