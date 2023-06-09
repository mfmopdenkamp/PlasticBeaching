import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="iteritems")
import load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample
import scipy.stats as stats
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from file_names import *

df = pd.read_csv(f'data/{file_name_4}.csv', parse_dates=['time_start', 'time_end'])

# normalize the data
for column in df.columns:
    if column not in ['time_start', 'time_end', 'beaching_flag']:
        df[column] = (df[column] - df[column].mean()) / df[column].std()

cor = df.corr(numeric_only=True)

y_column = 'beaching_flag'

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


def get_even_distribution(x_train, y_train):

    count_true = sum(y_train)
    count_false = len(y_train) - count_true

    count_min = min(count_true, count_false)

    x_true = x_train[y_train]
    x_false = x_train[~y_train]

    x_false_resampled = resample(x_false, n_samples=count_min, replace=False, random_state=42)

    x_train_resampled = np.concatenate((x_true, x_false_resampled), axis=0)
    y_train_resampled = np.concatenate((np.ones(count_min), np.zeros(count_min)), axis=0)

    return x_train_resampled, y_train_resampled


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, shuffle=False)
x_train_resampled, y_train_resampled = get_even_distribution(x_train, y_train)

best_params_1 = {'max_depth': 5, 'max_features': 'log2', 'min_samples_split': 2, 'n_estimators': 50}
param_grid = {'n_estimators':[10, 50, 100], 'min_samples_split':[2, 5, 10, 20], 'max_depth':[None, 5, 10, 20],
              'max_features':[None, 'sqrt', 'log2']}

# estimator = RandomForestClassifier()
# grid_search = HalvingGridSearchCV(estimator, param_grid=param_grid, verbose=2)
# grid_search.fit(x, y)
# df_gs = pd.DataFrame(grid_search.cv_results_)

estimator = RandomForestClassifier(**best_params_1)

estimator.fit(x_train_resampled, y_train_resampled)

y_pred = estimator.predict(x_test)

a_score = accuracy_score(y_test, y_pred)

c_matrix = confusion_matrix(y_test, y_pred)

# plot the best features
importances = estimator.feature_importances_
std = np.std([tree.feature_importances_ for tree in estimator.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
plt.figure()
plt.title("Feature importances")
plt.bar(range(x.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(x.shape[1]), x.columns[indices], rotation=90)
plt.xlim([-1, x.shape[1]])
plt.tight_layout()
plt.show()

