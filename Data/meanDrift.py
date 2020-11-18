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

path = 'I:/breakingIW/nonhydro/'

ds2 = xr.open_dataset(path + 'KeffThInterp.nc')
ds2.coords['time'] = np.linspace(0, 300*(tcount-1), tcount) / 3600

ds3 = xr.open_dataset(path + 'LagDiffTh.nc')
ds3.coords['time'] = np.linspace(300, 300*(tcount-3), tcount-3) / 3600

#%% calculation
keffGrd = ds2.nkeff.diff('new') / ds2['new'].diff('new') * 300 * 2e-5
drift   = ds3.drift.rolling(time=3, center=True,
                           min_periods=1).mean().rolling(Z=3, center=True,
                           min_periods=1).mean()

#%% plotting
import proplot as pplt

pplt.rc.update({'figure.facecolor': 'white'})

array = [
    [1],
    [2]
]

fig, ax = pplt.subplots(array, figsize=(6.5,8), sharey=3)

ax.format(abc=True, abcloc='l', abcstyle='(a)')

m1 = ax[0].contourf(keffGrd.T*12, cmap=plt.cm.RdBu_r, vmax=0.1, vmin=-0.08)
m2 = ax[1].contourf(drift.T, cmap=plt.cm.RdBu_r, vmax=0.1, vmin=-0.08)

fontsize = 15

ax[0].set_title('gradient of KWD', fontsize=fontsize)
a=ax[0].colorbar(m1, loc='r', width=0.15, row=2, extend='both')
a.ax.tick_params(labelsize=11)

ax[1].set_title('mean drift', fontsize=fontsize)
a=ax[1].colorbar(m2, loc='r', width=0.15)
a.ax.tick_params(labelsize=11)

for a in ax:
    a.set_ylabel('depth (m)', fontsize=fontsize)
    a.set_xlabel('time (hour)', fontsize=fontsize)

ax.axvline(13, color=(0.5,0.5,0.5), linestyle='--')
ax.axvline(68.5, color=(0.5,0.5,0.5), linestyle='--')
ax.axvline(75.5, color=(0.5,0.5,0.5), linestyle='--')
# ax.axvline(112, color='k', linestyle='--')
ax.set_xlim([0, 120])
ax.set_ylim([-200, 0])

