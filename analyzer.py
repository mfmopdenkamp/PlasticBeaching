import matplotlib.pyplot as plt
import numpy as np


def analyze_death_codes(ds, verbose=False):
    death_types = np.unique(ds.type_death)
    n_death_types = np.zeros(len(death_types))
    for i_death, death_type in enumerate(death_types):
        n_death_types[i_death] = sum(ds.type_death == death_type)

    if verbose:
        fig, ax = plt.subplots()
        ax.bar(death_types, n_death_types)
        ax.set_xlabel('death type')
        ax.set_ylabel('# drifters')
        plt.xticks(death_types)
        plt.show()

    return death_types, n_death_types


if __name__ == '__main__':
    import load_data
    ds = load_data.get_ds_drifters(proximity_of_coast=10)

    analyze_death_codes(ds, verbose=True)