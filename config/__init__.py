import tomli

with open('config.toml', 'rb') as f:
    config = tomli.load(f)

filename = config['name']
filename_wind = config['name_wind']
shoreline_resolution = config['shoreline_resolution']
