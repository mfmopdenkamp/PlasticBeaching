import load_data

ds = load_data.get_ds_drifters('gdp_v2.00.nc_no_sst')

with open('data_vars.txt', 'w') as f:
    for item in ds.data_vars.items():
        item = item[0]
        f.write(f'{item}\n{ds[item].attrs}\n\n')
