import tomli

with open('config/config.toml', 'rb') as f:
    config = tomli.load(f)

filename = config['filename']
filename_wind = config['filename_wind']
filename_wind_plus = config['filename_wind_plus']
shoreline_resolution = config['shoreline_resolution']
