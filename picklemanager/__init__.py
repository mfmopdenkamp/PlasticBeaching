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


def dump_pickle(obj, pickle_path=None, base_name=None):
    if pickle_path is None:
        pickle_path = create_pickle_path(('temp' if base_name is None else base_name))
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
        dump_pickle(obj, pickle_path)
    return obj


def create_pickle_path(base_name):
    return pickle_folder_path / f'{base_name}.pkl'


def create_pickle_ds_gdp_name(percentage=100, location_type=None, drogued=None,
                              max_aprox_distance_km=None, start_date=None, end_date=None,
                              min_aprox_distance_km=None, type_death=None, random_set=1):

    if isinstance(location_type, str):
        if location_type[0] in ['A', 'a']:  # Argos
            location_type = False
        elif location_type[0] in ['G', 'g']:  # GPS
            location_type = True

    return f'ds_gdp{(f"_{percentage}%_{random_set}" if percentage < 100 else "")}' \
           f'{("_" + str(start_date) if start_date is not None else "")}' \
           f'{("_" + str(end_date) if end_date is not None else "")}' \
           f'{("_gps" if location_type else ("" if location_type is None else "_argos"))}' \
           f'{("_drogued" if drogued else ("" if drogued is None else "_undrogued"))}' \
           f'{("__" + str(min_aprox_distance_km) + "km" if min_aprox_distance_km is not None else "")}' \
           f'{("_" + str(max_aprox_distance_km) + "km" if max_aprox_distance_km is not None else "")}' \
           f'{("_death" + str(type_death) if type_death is not None else "")}'
