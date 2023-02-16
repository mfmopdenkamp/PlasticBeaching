import os
import pickle

pickle_folder = 'pickledumps/'


def check_pickle_folder():
    if not os.path.exists(pickle_folder):
        os.mkdir(pickle_folder)
        print(f"Directory '{pickle_folder}' created.")


def load_pickle(pickle_name):
    check_pickle_folder()
    if os.path.isfile(pickle_name):
        print(f'Loading {pickle_name}... ', end='')
        with open(pickle_name, 'rb') as f:
            obj = pickle.load(f)
        print("Done")
        return obj
    else:
        raise FileNotFoundError('Pickle not found.')


def dump_pickle(pickle_name, obj):
    with open(pickle_name, 'wb') as f:
        pickle.dump(obj, f)


def pickle_wrapper(base_name, function, *args, **kwargs):
    pickle_name = create_pickle_name(base_name)
    check_pickle_folder()
    try:
        obj = load_pickle(pickle_name)
    except FileNotFoundError:
        obj = function(*args, **kwargs)
        with open(pickle_name, 'wb') as f:
            pickle.dump(obj, f)
        print(f'Object written to {pickle_name}.')
    return obj


def create_pickle_name(base_name):
    return pickle_folder + base_name + '.pkl'
