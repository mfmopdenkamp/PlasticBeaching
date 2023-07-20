import picklemanager as pickm
import matplotlib.pyplot as plt

filter_outliers = True

bar_heights = []
xtick_labels = []

a_majority = 0.9087378640776699
a_dist = 0.9009708737864077

bar_heights.append(a_majority)
xtick_labels.append('Majority')
bar_heights.append(a_dist)
xtick_labels.append('Single split')

remove_tidal = False
remove_directionality = False
remove_coastal_type = False
undersampling = False

a_tree = 0.9106796116504854
a_rf = 0.9184466019417475

bar_heights.append(a_tree)
xtick_labels.append('Tree - all')
bar_heights.append(a_rf)
xtick_labels.append('Random forest - all')


remove_tidal = True
remove_directionalit = True
remove_coastal_type = True
undersampling = False

a_tree = 0.9087378640776699
a_rf = 0.9087378640776699

bar_heights.append(a_tree)
xtick_labels.append('Tree\nMinimum features')
bar_heights.append(a_rf)
xtick_labels.append('Random forest\nMinimum features')

#%%
plt.bar(xtick_labels, bar_heights)
plt.ylabel('Accuracy score')
plt.ylim([0.9, 0.92])
plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig('figures/model_results.png', dpi=300, bbox_inches='tight')
plt.show()

#%%

bar_heights.append(a_tree)
xtick_labels.append('Tree\nMinimum features\nundersampling')
bar_heights.append(a_rf)
xtick_labels.append('Random forest\nMinimum features\nundersampling')


#%%
pickle_name = pickm.create_pickle_path(f'random_forest_results_{filter_outliers}_{remove_tidal}_{remove_directionality}_'
                                           f'{undersampling}_{remove_coastal_type}')
grid_search_rf = pickm.load_pickle(pickle_name)

pickle_name = pickm.create_pickle_path(f'decision_tree_{filter_outliers}_{remove_tidal}_{remove_directionality}_'
                                       f'{undersampling}_{remove_coastal_type}')
grid_search_tree = pickm.load_pickle(pickle_name)

#%% print the best hyperparameters
print(f'RF\t{grid_search_rf.best_params_}')

print(f'Single Tree\t{grid_search_tree.best_params_}')
