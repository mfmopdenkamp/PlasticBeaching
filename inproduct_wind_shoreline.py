import numpy as np
import pandas as pd
import load_data

df = load_data.get_subtrajs('events_prep_non_splitted_drogued.csv')

# check if the hypothenuse of de and dn is equal to the nearest distance to the coast
differences = np.empty(len(df))
for i, (de, dn, d) in enumerate(zip(df['de'], df['dn'], df['nearest shore'])):
    differences[i] = np.hypot(de, dn) - d
