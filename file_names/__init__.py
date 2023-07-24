import os

data_folder = 'data/'
file_name_1 = data_folder + 'drifter_state_24.csv'
file_name_2 = os.path.splitext(file_name_1)[0] + '_wind.csv'
file_name_3 = os.path.splitext(file_name_2)[0] + '_shore.csv'
file_name_4 = os.path.splitext(file_name_3)[0] + '_tidal.csv'

