# -*- coding: utf-8 -*-
'''
Created on 2020.06.19

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
'''

#%% load data
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


trName = 'THETA'
tcount = 1440

path = 'I:/breakingIW/nonHydro/diags/'

ds2 = xr.open_dataset(path + 'KeffThInterp_dec_gt2.nc')
ds2.coords['time'] = np.linspace(0, 300*(tcount-1), tcount) / 3600

ds3 = xr.open_dataset(path + 'LagDiffTh_dec_gt2.nc')
ds3.coords['time'] = np.linspace(300, 300*(tcount-3), tcount-3) / 3600

#%% calculate tracer change
# tr1 = ds.TRAC01.where(ds.TRAC01!=0).mean('XC').squeeze().load()
tr2 = ds2[trName]

diff = ((tr2).differentiate('time') * 1e4).rolling(time=9, center=True,
                                                   min_periods=9).mean()


#%% calculate Knum
import xmitgcm
from GeoApps.GridUtils import add_MITgcm_missing_metrics
from GeoApps.DiagnosticMethods import Dynamics


path2 = 'I:/breakingIW/nonHydro/'
ds = xmitgcm.open_mdsdataset(path2 + 'output/', grid_dir=path2,
                               delta_t=2, prefix=['Stat'])
dset, grid = add_MITgcm_missing_metrics(ds, periodic=['X'])

tr = dset[trName].where(dset[trName]!=0).astype(np.float64)

grdS = Dynamics(dset, grid).cal_squared_gradient(tr, dims=['X', 'Z'])

area    = grid.integrate(tr-tr+1, ['X', 'Z']).load()
trave   = grid.integrate(tr     , ['X', 'Z']).load()
tr2ave  = grid.integrate(tr**2  , ['X', 'Z']).load()
grdSave = grid.integrate(grdS   , ['X', 'Z']).load()
#%%
# a = tr2ave.diff('time') / 300
a = tr2ave.differentiate('time')
b = grdSave[1:]
Knum = (-a/b).squeeze().rolling(time=11, center=True, min_periods=1).mean()
# Knum[:3] = 0
Knum['time'] = np.linspace(300, 300*(tcount-1), tcount-1) / 3600

#%% plotting
import proplot as pplt

array = [
    [1],
    [2],
    [3]
]

fig, ax = pplt.subplots(array, figsize=(6.5,8), sharey=3)


ax.format(abc=True, abcloc='l', abcstyle='(a)')

lagDff = ds3.diffu.rolling(time=1, center=True,
                           min_periods=1).mean().rolling(Z=1, center=True,
                           min_periods=1).mean()

m1 = ax[2].pcolormesh(diff.T, cmap=plt.cm.RdBu_r, levels=np.linspace(-9,9,19))
m2 = ax[0].pcolormesh(ds2.nkeff.T.rolling(new=3, center=True, min_periods=1).mean(),
                      cmap=plt.cm.RdYlBu_r, levels=np.linspace(0, 12, 25))
m3 = ax[1].pcolormesh((lagDff.T/Knum).rolling(time=9, center=True, min_periods=1).mean(),
                      cmap=plt.cm.RdYlBu_r,
                      levels=np.linspace(0, 12, 25))

CS = ax[2].contour(ds2[trName].T,
              levels=[25.73, 25.74, 25.75, 25.76, 25.77, 25.78, 25.79, 25.80, 25.82,
                      25.85, 25.90, 25.95, 26, 26.05, 26.1, 26.15, 26.18, 26.2,
                      26.22],
              colors='k')
ax[2].clabel(CS, CS.levels, inline=True, fontsize=12, fmt='%.2f')

fontsize = 15

ax[2].set_title('temporal changes of $\\theta$ in Z-coordinate', fontsize=fontsize)
ax[2].set_xticks([0, 24, 48, 72, 96, 120])
a=ax[2].colorbar(m1, loc='r', width=0.15, ticks=[-9, -6, -3, 0, 3, 6, 9])
a.ax.tick_params(labelsize=11)

ax[0].set_title('normalized $K_\\rho$ in Z-coordinate', fontsize=fontsize)
a=ax[0].colorbar(m2, loc='r', width=0.15, row=2, ticks=[0, 3, 6, 9, 12])
a.ax.tick_params(labelsize=11)

ax[1].set_title('normalized $K_L$ in Z-coordinate', fontsize=fontsize)
a=ax[1].colorbar(m3, loc='r', width=0.15, ticks=[0, 3, 6, 9, 12])
a.ax.tick_params(labelsize=11)

for a in ax:
    a.set_ylabel('depth (m)', fontsize=fontsize)
    a.set_xlabel('time (hour)', fontsize=fontsize)

# ax[0].contour(ds2[trName].T, colors='k', levels=[25.774, 25.807, 25.822])
# ax[1].contour(ds2[trName].T, colors='k', levels=[25.774, 25.807, 25.822])
# ax[0].contour(ds2[trName].T, colors='k', levels=[25.81, 25.82, 25.825])
# ax[1].contour(ds2[trName].T, colors='k', levels=[25.81, 25.82, 25.825])

ax.axvline(13, color=(0.2,0.7,0.2), linestyle='--')
ax.axvline(61, color=(0.2,0.7,0.2), linestyle='--')
ax.axvline(105, color=(0.2,0.7,0.2), linestyle='--')
# ax.axvline(112, color='k', linestyle='--')
ax.set_xlim([0, 120])

