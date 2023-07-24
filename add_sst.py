import load_data
import picklemanager as pickm

ds = pickm.load_pickle(pickm.create_pickle_path('gdp_drogue_presence'))

ds_sst = pickm.load_pickle(pickm.create_pickle_path('gdp_v2.00.nc'))

ds_final = ds.merge(ds_sst)

print(ds_final.info())

pickm.dump_pickle(ds_final, base_name='gdp_drogue_presence_sst')