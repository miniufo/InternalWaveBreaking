# -*- coding: utf-8 -*-
'''
Created on 2020.06.19

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
'''

#%% load data
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
import xmitgcm
import xarray as xr
from utils.PlotUtils import plot


trName = 'TRAC01'
tcount = 1440

path = 'I:/breakingIW/nonhydro/'

ds = xmitgcm.open_mdsdataset(data_dir=path+'output/',
                             grid_dir=path,
                             delta_t=2,
                             prefix=['Stat'])
ds.coords['time'] = ds.coords['time'] / 3600

ds2 = xr.open_dataset(path + 'keff.nc')
ds2.coords['time'] = ds2.coords['time'] / 3600

#%% calculation

tr1 = ds.TRAC01.where(ds.TRAC01!=0).mean('XC').squeeze().load()
tr2 = ds2.TRAC01


#%% plotting
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,8))

plot(tr1, ax=ax[0], ptype='contour', y='Z',
     clevs=np.linspace(25.7, 26.2, 21), ylint=25, xlint=12)
plot(tr2, ax=ax[1], ptype='contour', y='new',
     clevs=np.linspace(25.7, 26.2, 21), ylint=25, xlint=12)


ax[0].set_title('theta evolution in x-z coordinate', x=0.5, y=1, fontsize=15)
ax[0].set_xlabel('', fontsize=13)
ax[0].set_ylim([-200, 0])
ax[0].set_ylabel('depth (m)', fontsize=13)
ax[0].tick_params(axis='both', labelsize=12, rotation=0)
ax[1].set_title('theta evolution in X-Z coordinate', x=0.5, y=1, fontsize=15)
ax[1].set_xlabel('horizontal distance (m)', fontsize=13)
ax[1].set_ylabel('depth (m)', fontsize=13)
ax[1].tick_params(axis='both', labelsize=12, rotation=0)

plt.tight_layout()

#%% plotting
import proplot as pplt

pplt.rc.update({'figure.facecolor': 'white', 'fontsize': 17})

array = [
    [1],
    [2]
]

fig, ax = pplt.subplots(array, figsize=(7,6))

m1 = tr1.plot(ax=ax[0], add_colorbar=False, cmap=plt.cm.jet, y='Z',
                      vmin=25.75, vmax=26.25)
tr2.plot(ax=ax[1], add_colorbar=False, cmap=plt.cm.jet, y='new',
                 vmin=25.75, vmax=26.25)


ax.format(abc=True, abcloc='l', abcstyle='(a)')


ax[0].format(title='theta evolution in x-z coordinate',
             ylabel='depth (m)',
             xlabel='time (hour)')

ax[1].format(title='theta evolution in X-Z coordinate',
             ylabel='depth (m)',
             xlabel='time (hour)')
ax[1].colorbar(m1, loc='bottom', width=0.15)

