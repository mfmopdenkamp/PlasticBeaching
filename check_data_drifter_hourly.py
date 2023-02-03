import matplotlib.pyplot as plt
import cartopy.crs as crs
import pickle_manager as pickm
import load_data


filename = 'gdp_v2.00.nc'
ds = pickm.load_pickle_wrapper(filename, load_data.drifter_data_hourly, filename)

ds_subset = pickm.load_pickle('pickledumps/gdp_v2.00.ncsubset_10km.pkl')

fig2 = plt.figure(dpi=300)
ax = plt.axes(projection=crs.PlateCarree())

# ax.scatter(ds.longitude.values, ds.latitude.values, color='r', s=3,
#            transform=crs.PlateCarree(), label='all')
ax.scatter(ds_subset.longitude.values, ds_subset.latitude.values, color='b', s=3,
           transform=crs.PlateCarree(), label='within 10km')

ax.coastlines()
ax.legend()
plt.show()

print(ds.dims)

print(ds.data_vars)

print(ds.attrs)

print(ds.obs.shape[0])

