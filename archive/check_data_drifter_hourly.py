import matplotlib.pyplot as plt
import cartopy.crs as crs
import numpy as np
import picklemanager as pickm
import load_data

file_name = pickm.create_pickle_path('gdp')
ds = load_data.get_ds_drifters()

fig2 = plt.figure(dpi=300)
ax = plt.axes(projection=crs.PlateCarree())

# ax.scatter(ds.longitude.values, ds.latitude.values, color='r', s=3,
#            transform=crs.PlateCarree(), label='all')
ax.scatter(ds.longitude.values, ds.latitude.values, color='b', s=3,
           transform=crs.PlateCarree(), label='within 10km')

ax.coastlines()
ax.legend()
plt.show()

print(ds.dims)

print(ds.data_vars)

print(ds.attrs)

print(ds.obs.shape[0])

years = ds.end_date.values.astype('datetime64[Y]').astype(int) + 1970
for i, (year, ed) in enumerate(zip(years, ds.end_date.values)):
    if year > 2020:
        print(ed)

death_type, dt_counts = np.unique(ds.type_death.values,
                                  return_counts=True)
plt.figure()
plt.bar(death_type, dt_counts)
plt.xlabel('Death Type')
plt.ylabel('Count')
plt.show()
