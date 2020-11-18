# -*- coding: utf-8 -*-
'''
Created on 2020.06.19

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
'''
#%% Get IDs
import xarray as xr
import sys
import xmitgcm
sys.path.append('../../floater/')
from GeoApps.GridUtils import add_MITgcm_missing_metrics
from utils.IOUtils import read_flt_bcolz

# folder = 'I:/cartRL_advSchemes/FLTReadTest/'
folder = 'I:/breakingIW/nonHydro/'

# npart, time, x, y, z, i, j, k, p, u, v, t, s
fset = read_flt_bcolz(folder + '.bcolz',
                    cond="((x>=996) & (x<=1004)) & (time==300) & (s!=0)",
                    # cond="(npart>10000000) & (npart<=10001000) & (time>0)",
                    fields=['npart', 'time', 'x', 's'],
                    out='xarray').dropna('npart')

#%% Read FLT by IDs
ids = ['(npart=='+str(npart.values)+')' for npart in fset.npart]

cond = ' | '.join(ids)

fset = read_flt_bcolz(folder + '.bcolz',
                    cond=cond,
                    fields=['npart', 'time', 'x', 'z', 's'],
                    out='xarray').dropna('npart')

#%% Read Tracer
import numpy as np

dset = xmitgcm.open_mdsdataset(data_dir=folder+'output/', grid_dir=folder,
                               prefix=['Stat'],delta_t=2)

dset, grid = add_MITgcm_missing_metrics(dset, periodic='X')

mask = dset['TRAC01'] != 0

tcount = 1440
ds2 = xr.open_dataset(folder + 'KeffThInterp.nc')
ds2.coords['time'] = np.linspace(0, 300*(tcount-1), tcount).astype('int')

#%% interpolate
from GeoApps.ArrayUtils import interp1d

fset2 = fset.isel(time=slice(0,-1))

xpos = fset2.x
zpos = fset2.z
time = fset2.time

xpos, time = xr.broadcast(xpos, time)

var1 = dset['THETA'].where(mask).squeeze().load()
trajThe = var1.interp({'XC':xpos, 'Z':zpos, 'time':time})
trajThe_nonan = trajThe.dropna('npart')

def interp_coords(trajThe, TRAC01):
    interpDim = 'new'
    coord = TRAC01[interpDim]
    
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

    return varIntp

trajZeq = interp_coords(trajThe, ds2.THETA.isel(time=slice(1,tcount)))
trajZeq_nonan = trajZeq.dropna('npart')

id_nan = trajZeq.npart.where(xr.ufuncs.isnan(trajZeq.mean('time', skipna=False)),
                             drop=True)

trajZeq_nan = trajZeq.sel(npart=id_nan)

# tmp = trajThe.where(xr.ufuncs.logical_and(fset.s[:,0]<20,fset.s[:,0]>18))

# trajZeq = trajZeq.rename('Zeq_all').reset_coords(drop=True)
# trajZeq_nonan = trajZeq_nonan.rename('Zeq_nona').reset_coords(drop=True)
# trajZeq_nan = trajZeq_nan.rename('Zeq_hasna').reset_coords(drop=True)

# ds_traj = xr.merge([trajZeq, trajZeq_nonan, trajZeq_nan])

# ds_traj.to_netcdf(folder + 'traj.nc')


#%%
import matplotlib.pyplot as plt
import proplot as pplt


pplt.rc.update({'figure.facecolor': 'white'})

fig, ax = pplt.subplots(nrows=3, ncols=1, figsize=(5.5,8), sharey=0)

skip = 18
tlen = 1439

zpos_nonan = zpos.sel(npart=trajZeq_nonan.npart)

vtmp = zpos_nonan[::skip].copy()
vtmp.coords['time'] = np.linspace(300/3600, 300*tlen/3600, tlen)
vtmp.plot(ax=ax[0], hue='npart', lw=1.6, add_legend=False, yincrease=True)
vtmp = trajThe_nonan[::skip].copy()
vtmp.coords['time'] = np.linspace(300/3600, 300*tlen/3600, tlen)
vtmp.plot(ax=ax[1], hue='npart', lw=1.6, add_legend=False, yincrease=True)
vtmp = trajZeq_nonan[::skip].copy()
vtmp.coords['time'] = np.linspace(300/3600, 300*tlen/3600, tlen)
vtmp.plot(ax=ax[2], hue='npart', lw=1.6, add_legend=False, yincrease=True)

ax.format(
    abc=True, abcloc='l', abcstyle='(a)',
)

ax[0].set_title('original z-coordinates', fontsize=14)
ax[0].set_ylabel('depth (m)', fontsize=12)
ax[0].set_xlim([0, 120])
ax[0].set_xticks([0, 24, 48, 72, 96, 120])
ax[0].set_ylim([-200, 0])

ax[1].set_title('contour coordinates', fontsize=14)
ax[1].set_ylabel('theta (°C)', fontsize=12)
ax[1].set_xlim([0, 120])
ax[1].set_xticks([0, 24, 48, 72, 96, 120])
ax[1].set_ylim([25.75, 26.25])

ax[2].set_title('transformed Z-coordinates', fontsize=14)
ax[2].set_ylabel('depth (m)', fontsize=12)
ax[2].set_ylim([-200, 0])
ax[2].set_xlim([0, 120])
ax[2].set_xticks([0, 24, 48, 72, 96, 120])
ax[2].set_xlabel('time (hour)', fontsize=12)

