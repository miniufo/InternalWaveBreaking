# -*- coding: utf-8 -*-
"""
Created on 2020.06.23

@author: MiniUFO
"""
#%% read_fields
import xmitgcm

path = 'I:/breakingIW/nonHydro/'

dset = xmitgcm.open_mdsdataset(data_dir=path+'output/',
                               grid_dir=path,
                               prefix=['Stat'], delta_t=2)

#%%
import xarray as xr
import numpy as np
from GeoApps.ArrayUtils import interp1d

tcount = len(dset.time)

dsKeff = xr.open_dataset(path + 'KeffTrInterp.nc')

dsKeff.coords['time'] = np.linspace(0, 300*(tcount-1), tcount).astype('int')
dsKeff = dsKeff.rename({'new':'Z'})


def interp_coords(trajThe, TRAC01):
    interpDim = 'Z'
    coord = TRAC01[interpDim]
    
    trajThe = trajThe.rename({'Z':'tmp'})
    
    increasing = False
    
    coord, TRAC01 = xr.broadcast(coord, TRAC01)
    
    varIntp = xr.apply_ufunc(interp1d, trajThe, TRAC01, coord,
              kwargs={'inc': increasing},
              dask='allowed',
              input_core_dims =[[], [interpDim], [interpDim]],
              # output_core_dims=[[interpDim]],
              exclude_dims=set((interpDim,)),
              vectorize=True
              ).rename('zEq')

    return varIntp.rename({'tmp':'Z'})

# zpos = interp_coords(dset.TRAC01, dsKeff.TRAC01)

re = []

for tim in dset.time:
    if (tim/300) % 50 == 0:
        print(tim.values)
    tr_xz = (dset.TRAC01.where(dset.TRAC01!=0)).squeeze().sel(time=tim)
    tr_XZ = dsKeff.TRAC01.squeeze().sel(time=tim)
    ke_XZ = dsKeff.nkeff.squeeze().sel(time=tim)
    ke_XZ[np.isnan(ke_XZ)] = 1
    
    zpos  = interp_coords(tr_xz, tr_XZ)
    mask = np.isnan(zpos)
    zpos.values[mask.values] = 1
    ke_xz = ke_XZ.interp(coords={'Z':zpos})
    ke_xz.values[mask.values] = np.nan
    re.append(ke_xz)

nkeff = xr.concat(re, 'time').to_netcdf(path + 'nkeffxz.nc')

kmean = nkeff.mean('time')
kprof = kmean.mean('XC')

#%% animate
from utils.PlotUtils import animate

ani=animate(nkeff, vmin=0, vmax=11)

ani.save(path+'nkeffxz.gif', writer='imagemagick', fps=12)

#%% time-mean
import matplotlib.pyplot as plt
import proplot as pplt

pplt.rc.update({'figure.facecolor': 'white'})
pplt.rc['axesfacecolor'] = 'white'

array = [
    [1, 1, 2]
]

fig, ax = pplt.subplots(array, figsize=(7,4), sharex=0)

ax.format(abc=True, abcloc='l', abcstyle='(a)')

m1 = ax[0].contourf(kmean, cmap=plt.cm.jet,
               levels=pplt.arange(1,6,0.05), add_colorbar=False)
ax[0].colorbar(m1, loc='bottom', width=0.1)
ax[0].format(title='time mean $K_{WD}$',
             xlabel='x-distance (m)',
             ylabel='depth (m)')

kprof.plot(ax=ax[1], y='Z')
ax[1].format(title='x-mean of $K_{WD}$',
             xlabel='$K_{WD}$ ($m^2$ $s^{-1}$)',
             ylabel='depth (m)')

