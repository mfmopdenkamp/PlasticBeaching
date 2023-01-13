import pandas as pd
import load_data
import pickle_manager as pickm

filename = 'dist2coast.txt.bz2'
df_d2s = pickm.load_pickle_wrapper(f'{filename}', pd.read_csv, load_data.data_folder+filename,
                                   delim_whitespace=True, names=['longitude', 'latitude', 'distance'],
                                   header=None, compression='bz2')
