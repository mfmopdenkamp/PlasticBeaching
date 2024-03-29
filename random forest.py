import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="iteritems")
import picklemanager as pickm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils import resample
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from file_names import *
import toolbox as tb

y_column = 'beaching_flag'

df = pd.read_csv(file_name_4, parse_dates=['time_start', 'time_end'], index_col='ID')
df.drop(columns=['ID.2', 'ID.1', 'longitude_start', 'latitude_start', 'longitude_end', 'latitude_end',
                 ], inplace=True, errors='ignore')
df['speed'] = np.hypot(df['velocity_north'], df['velocity_east'])
df.rename(columns={'velocity_north': 'velocity_v', 'velocity_east': 'velocity_u',
                   'shortest_distance_n': 'shore_distance_y', 'shortest_distance_e': 'shore_distance_x',
                   }, inplace=True)
# Renaming columns based on the mentioned conditions
new_columns = {col: col.replace('score_', 'scr_').replace('deg', '').replace('km', '')
               for col in df.columns if col.startswith('score')}

# Rename the DataFrame columns
df.rename(columns=new_columns, inplace=True)

df['inprod_u10m_d'] = tb.get_unit_inproducts(df['shore_distance_x'].values,
                                                       df['shore_distance_y'].values,
                                                         df['wind10m_u_mean'].values,
                                                         df['wind10m_v_mean'].values,
                                                       unit=False)
df['inprod_unit_u10m_d'] = tb.get_unit_inproducts(df['shore_distance_x'].values,
                                                            df['shore_distance_y'].values,
                                                            df['wind10m_u_mean'].values,
                                                            df['wind10m_v_mean'].values)
df['inprod_v_d'] = tb.get_unit_inproducts(df['shore_distance_x'].values, df['shore_distance_y'].values,
                                                        df['velocity_u'].values, df['velocity_v'].values,
                                                        unit=False)
df['inprod_unit_v_d'] = tb.get_unit_inproducts(df['shore_distance_x'].values, df['shore_distance_y'].values,
                                                        df['velocity_u'].values, df['velocity_v'].values)

filter_outliers = True
remove_tidal = False
remove_directionality = False
remove_coastal_type = False
undersampling = False


#%% filter out drifters with more than 15 false beaching flags and more than 6 true beaching flags
if filter_outliers:
    table_beaching_per_drifter = df.groupby('drifter_id').beaching_flag.value_counts().unstack().fillna(0).astype(int)
    drifter_ids_to_keep = table_beaching_per_drifter[
        (table_beaching_per_drifter[False] <= 15) & (table_beaching_per_drifter[True] <= 6)].index
    df = df[df.drifter_id.isin(drifter_ids_to_keep)]


#%%
if remove_tidal:
    df['total_tidal'] = np.zeros(len(df))
    tidal_columns = []
    for column in df.columns:
        if column[-18:] == 'tidal_elevation_mm':
            df['total_tidal'] += np.abs(df[column])
            tidal_columns.append(column)

    df.drop(columns=tidal_columns, inplace=True, errors='ignore')

if remove_directionality:
    df.drop(columns=['velocity_north', 'velocity_east', 'shortest_distance_n', 'shortest_distance_e', 'wind_10m_v_min',
                     'wind_10m_v_max', 'wind_10m_v_mean', 'wind_10m_v_std', 'wind_10m_u_min', 'wind_10m_u_max'],
            inplace=True, errors='ignore')

if remove_coastal_type:
    coastal_type_columns = []
    for column in df.columns:
        try:
            if column.split('_')[1] in ['beach', 'bedrock', 'wetland']:
                coastal_type_columns.append(column)
        except IndexError:
            pass
    df.drop(columns=coastal_type_columns, inplace=True, errors='ignore')

df.drop(columns=['drifter_id'], inplace=True, errors='ignore')
#%%
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
    # if the input has been a dataframe, convert back to dataframe
    if isinstance(x_train, pd.DataFrame):
        x_train_resampled = pd.DataFrame(x_train_resampled, columns=x_train.columns)
    return x_train_resampled, y_train_resampled


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, shuffle=False)
if undersampling:
    x_train, y_train = get_even_distribution(x_train, y_train)
#%% give results for predicting all false
y_pred_all_false = np.zeros(len(y_test))
a_score_all_false = accuracy_score(y_test, y_pred_all_false)
p_score_all_false = precision_score(y_test, y_pred_all_false)
r_score_all_false = recall_score(y_test, y_pred_all_false)
f1_score_all_false = f1_score(y_test, y_pred_all_false)
c_matrix_all_false = confusion_matrix(y_test, y_pred_all_false)


# %% base line model
from toolbox import hard_coded_exp_fit

y_pred_base = hard_coded_exp_fit(x_test['shortest_distance'])
a_score_base = accuracy_score(y_test, y_pred_base)
p_score_base = precision_score(y_test, y_pred_base)
r_score_base = recall_score(y_test, y_pred_base)
f1_score_base = f1_score(y_test, y_pred_base)
c_matrix_base = confusion_matrix(y_test, y_pred_base)

# %% decision tree
pickle_name = pickm.create_pickle_path(f'decision_tree_new_{filter_outliers}_{remove_tidal}_{remove_directionality}_'
                                       f'{undersampling}_{remove_coastal_type}')
try:
    grid_search_tree = pickm.load_pickle(pickle_name)
except FileNotFoundError:
    params = {'splitter': ['best', 'random'], 'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10, 20, 50, 100], 'min_samples_leaf': [1, 2, 5, 10, 20],
                'max_features': [None, 'sqrt', 'log2']}
    grid_search_tree = GridSearchCV(DecisionTreeClassifier(), param_grid=params, verbose=1)
    grid_search_tree.fit(x_train, y_train)

    pickm.dump_pickle(grid_search_tree, pickle_name)

y_pred_tree = grid_search_tree.predict(x_test)
a_score_tree = accuracy_score(y_test, y_pred_tree)
a_score_tree_train = accuracy_score(y_train, grid_search_tree.predict(x_train))
p_score_tree = precision_score(y_test, y_pred_tree)
r_score_tree = recall_score(y_test, y_pred_tree)
f1_score_tree = f1_score(y_test, y_pred_tree)
c_matrix_tree = confusion_matrix(y_test, y_pred_tree)


#%% random forest
try:
    pickle_name = pickm.create_pickle_path(f'random_forest_new_{filter_outliers}_{remove_tidal}_{remove_directionality}_'
                                           f'{undersampling}_{remove_coastal_type}')
    grid_search_rf = pickm.load_pickle(pickle_name)
except FileNotFoundError:
    param_grid = {'n_estimators':[50, 100, 200], 'min_samples_split':[10, 20, 40, 80], 'max_depth':[None, 5, 10, 20],
                  'max_features':[None, 'sqrt', 'log2'], 'min_samples_leaf': [1, 2, 5, 10, 20]}
    grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, verbose=1)
    grid_search_rf.fit(x_train, y_train)

    pickm.dump_pickle(grid_search_rf, pickle_name)

# ===================>>>>
# best_params_1 = {'max_depth': None, 'max_features': None, 'min_samples_leaf': 10, 'min_samples_split': 20, 'n_estimators': 100}

y_pred_rf = grid_search_rf.predict(x_test)
a_score_rf = accuracy_score(y_test, y_pred_rf)
a_score_rf_train = accuracy_score(y_train, grid_search_rf.predict(x_train))
p_score_rf = precision_score(y_test, y_pred_rf)
r_score_rf = recall_score(y_test, y_pred_rf)
f1_score_rf = f1_score(y_test, y_pred_rf)
c_matrix_rf = confusion_matrix(y_test, y_pred_rf)
#%%
metrics = {'Model': ['Majority Class', '1-Split', 'Decision Tree', 'Random Forest'],
           'Accuracy': [a_score_all_false, a_score_base, a_score_tree, a_score_rf],
           'Precision': [p_score_all_false, p_score_base, p_score_tree, p_score_rf],
           'Recall': [r_score_all_false, r_score_base, r_score_tree, r_score_rf],
           'F1': [f1_score_all_false, f1_score_base, f1_score_tree, f1_score_rf]}

# Create dataframe
df = pd.DataFrame(metrics)
# Set index
df.set_index('Model', inplace=True)

x=np.arange(len(df.index))  # the label locations
width = 1/len(df.columns)-0.05  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

# Plot bar chart for Accuracy
ax.bar(x-1.5*width, df['Accuracy'], color='blue', width=width, label='Accuracy')

# Set y-axis limits
ax.set_ylim([min(df['Accuracy'])*0.99, max(df['Accuracy'])*1.01])
# Set y-axis label
ax.set_ylabel('Accuracy')
# Create twin axis
ax2 = ax.twinx()
# Plot other metrics (drop Accuracy column before plotting)
ax2.bar(x-0.5*width, df['Precision'], color='red', width=width, label='Precision')
ax2.bar(x+0.5*width, df['Recall'], color='green', width=width, label='Recall')
ax2.bar(x+1.5*width, df['F1'], color='orange', width=width, label='F1')

ax.set_xticks(x)
ax.set_xticklabels(df.index)

ax.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.savefig('figures/model_results_all.png', dpi=300, bbox_inches='tight')
plt.show()


#%% write scores to file
with open('models_results_new.txt', 'a') as f:
    f.write(f'filter_outliers = {filter_outliers}\n remove_tidal = {remove_tidal}\n remove_directionality = '
            f'{remove_directionality}\n remove_coastal_type = {remove_coastal_type}\n undersampling = {undersampling}\n')
    f.write(f'a_majority = {a_score_all_false}\n')
    f.write(f'a_dist = {a_score_base}\n')
    f.write(f'a_tree = {a_score_tree}\n')
    f.write(f'a_rf = {a_score_rf}\n')
    f.write('\n')
    f.write(f'p_majority = {p_score_all_false}\n')
    f.write(f'p_dist = {p_score_base}\n')
    f.write(f'p_tree = {p_score_tree}\n')
    f.write(f'p_rf = {p_score_rf}\n')
    f.write('\n')
    f.write(f'r_majority = {r_score_all_false}\n')
    f.write(f'r_dist = {r_score_base}\n')
    f.write(f'r_tree = {r_score_tree}\n')
    f.write(f'r_rf = {r_score_rf}\n')
    f.write('\n')
    f.write(f'f1_majority = {f1_score_all_false}\n')
    f.write(f'f1_dist = {f1_score_base}\n')
    f.write(f'f1_tree = {f1_score_tree}\n')
    f.write(f'f1_rf = {f1_score_rf}\n')
    f.write('\n')
    f.write(f'a_tree_train = {a_score_tree_train}\n')
    f.write(f'a_rf_train = {a_score_rf_train}\n')
    f.write('\n')
    f.write(f'tree params = {grid_search_tree.best_params_}\n')
    f.write(f'rf params = {grid_search_rf.best_params_}\n')
    f.write('\n')
    f.write(f'All false:\n\n {c_matrix_all_false}\n')
    f.write(f'Base line:\n {c_matrix_base}\n')
    f.write(f'Decision tree:\n {c_matrix_tree}\n')
    f.write(f'Random forest:\n {c_matrix_rf}\n')
    f.write('\n')


#%% plot the best features
def plot_feature_importances(importances):
    top = x.shape[1] // 2
    indices = np.argsort(importances)

    plt.figure(figsize=(7, 5))
    plt.barh(range(top), importances[indices][-top:], align="center")
    plt.yticks(range(top), x.columns[indices][-top:])
    plt.ylim([-1, top])
    plt.tight_layout()

    plt.savefig('figures/feature_importances_grounding.png', dpi=300)

    plt.show()


plot_feature_importances(grid_search_rf.best_estimator_.feature_importances_)

#%% PDP plots
from sklearn.inspection import PartialDependenceDisplay

# Let's say 'rf' is your trained RandomForestClassifier, and 'X' is your features DataFrame

pdp_display = PartialDependenceDisplay.from_estimator(
    grid_search_rf.best_estimator_,                # trained model
    x_train,                 # features
    features=['speed', 'shore_distance']  # features to plot
)

pdp_display.plot()

plt.savefig('figures/pdp_velocity.png', dpi=300)
plt.show()


pdp_display = PartialDependenceDisplay.from_estimator(
    grid_search_rf.best_estimator_,                # trained model
    x_train,                 # features
    features=['shore_distance']  # features to plot
)

pdp_display.plot()

plt.savefig('figures/pdp_shortest_distance.png', dpi=300)
plt.show()

