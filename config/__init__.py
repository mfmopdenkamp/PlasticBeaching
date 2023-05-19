import tomli

with open('config/config.toml', 'rb') as f:
    config = tomli.load(f)

csv_wind = config['csv_wind']
csv_wind2 = config['csv_wind2']
csv_wind2_tides = config['csv_wind2_tides']

shoreline_resolution = config['shoreline_resolution']