import pathlib
import pickle

pickle_folder_name = 'pickledumps/'
pickle_folder_path = pathlib.Path(__file__).parent / pickle_folder_name


def check_pickle_folder():
    if not pathlib.Path.exists(pickle_folder_path):
        pathlib.Path.mkdir(pickle_folder_path)
        print(f"Directory '{pickle_folder_path}' created.")


def load_pickle(pickle_path):
    check_pickle_folder()
    if pathlib.Path.exists(pickle_path):
        print(f'Loading {pickle_path}... ', end='')
        with pickle_path.open(mode='rb') as f:
            obj = pickle.load(f)
        print("Done.")
        return obj
    else:
        raise FileNotFoundError('Pickle not found.')


def dump_pickle(pickle_path, obj):
    with pickle_path.open(mode='wb') as f:
        pickle.dump(obj, f)
    print(f'Object written to {pickle_path}.')


def pickle_wrapper(base_name, function, *args, **kwargs):
    pickle_path = create_pickle_path(base_name)
    check_pickle_folder()
    try:
        obj = load_pickle(pickle_path)
    except FileNotFoundError:
        obj = function(*args, **kwargs)
        dump_pickle(pickle_path, obj)
    return obj


def create_pickle_path(base_name):
    return pickle_folder_path / f'{base_name}.pkl'
