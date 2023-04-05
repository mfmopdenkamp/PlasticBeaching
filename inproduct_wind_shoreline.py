import numpy as np
import pandas as pd
import load_data

df = load_data.get_subtrajs('events_prep_non_splitted_drogued.csv')

# check if the hypothenuse of de and dn is equal to the nearest distance to the coast
differences = np.empty(len(df))
inproducts = np.empty(len(df))
for i, (de, dn, d, u, v) in enumerate(zip(df['de'], df['dn'], df['nearest shore'], df['u_mean'], df['v_mean'])):
    differences[i] = np.hypot(de, dn) - d
    inproducts[i] = de/d*u + dn/d*v
