# -*- coding: utf-8 -*-
"""
Created on 2020.06.23

@author: MiniUFO
"""
#%% read_fields
import xmitgcm
import xarray as xr
import numpy as np

path = 'I:/breakingIW/nonHydro/'

dset = xmitgcm.open_mdsdataset(data_dir=path+'output/',
                               grid_dir=path,
                               prefix=['Stat'], delta_t=2)

bset = xr.open_dataset(path + '/diags/binning/binLagTh_dec_gt2.nc')

#%%
delY = bset['sumD']/bset['cntD']

dYsmth = delY.rolling(time=3,
                      center=True,
                      min_periods=1).mean().rolling(x_bin=1,center=True,
                                                    min_periods=1).mean()
tr = dset.THETA.where(dset.TRAC01!=0)

grd = np.sqrt(tr.differentiate('XC')**2+tr.differentiate('Z')**2)


#%%
import proplot as pplt
import matplotlib.pyplot as plt

contours = np.array([25.81, 25.825, 25.82])
times = np.array([13, 61, 104]) * 3600

fig, ax = pplt.subplots(nrows=3, ncols=2, figsize=(11,8), sharey=3,
                        wspace=(0.1), hspace=(0.3,0.3))

ax.format(abc=True, abcloc='l', abcstyle='(a)')

for i, (ctr, tim) in enumerate(zip(contours, times)):
    m1 = ax[i*2].contourf(tr.sel(time=tim).squeeze(), cmap=plt.cm.jet,
                          levels=np.linspace(25.74, 26.25, 20), extend='both')
    m2 = ax[i*2+1].contourf(np.log(dYsmth.sel(time=tim-300)), cmap=plt.cm.jet,
                          levels=np.linspace(-9, -2, 29), extend='both')
    # m2 = ax[i*2+1].contourf(dYsmth.sel(time=tim-300), cmap=plt.cm.jet,
    #                       levels=np.linspace(0, 0.05, 33), extend='both')
    
    ax[i*2].contour(tr.sel(time=tim).squeeze(), levels=[ctr], colors='k')
    
    ax[i*2].format(title='$\\theta$ at {} hr'.format(tim/3600),
                   ylabel='depth (m)',
                   xlabel='x-distance (m)')
    ax[i*2].set_xlim([0, 8990])
    ax[i*2].set_ylim([-200, 0])
    
    ax[i*2+1].format(title='local disp. rate at {} hr'.format(tim/3600),
                     ylabel='depth (m)',
                     xlabel='x-distance (m)')
    ax[i*2+1].set_xlim([0, 8990])
    ax[i*2+1].set_ylim([-200, 0])

ax[4].colorbar(m1, loc='b', width=0.15, ticks=0.1)
ax[5].colorbar(m2, loc='b', width=0.15, ticks=1)

for a in ax:
    a.set_ylabel('depth (m)', fontsize=13)
    a.set_xlabel('x-distance (m)', fontsize=13)

#%%
import proplot as pplt
import matplotlib.pyplot as plt

contours = np.array([25.775, 25.796, 25.82])
times = np.array([13, 54.5, 75.5]) * 3600

fig, ax = pplt.subplots(nrows=3, ncols=3, figsize=(11,8), sharey=1,
                        wspace=(0.1, 0.1), hspace=(0.3, 0.3))

ax.format(abc=True, abcloc='ul', abcstyle='(a)',
          collabels=['isopycnals ($\\theta$)',
                     'local diapycnal dispersion rate',
                     'tracer gradient'])

for i, (ctr, tim) in enumerate(zip(contours, times)):
    m1 = ax[i*3].contourf(tr.sel(time=tim).squeeze(), cmap=plt.cm.jet,
                          levels=np.linspace(25.74, 26.25, 20), extend='both')
    m2 = ax[i*3+1].contourf(np.log(dYsmth.sel(time=tim-300)), cmap=plt.cm.jet,
                          levels=np.linspace(-9, -2, 33), extend='both')
    m3 = ax[i*3+2].contourf(grd.sel(time=tim).squeeze()*1e3, cmap=plt.cm.jet,
                          levels=np.linspace(0, 40, 41), extend='both')
    
    ax[i*3].contour(tr.sel(time=tim).squeeze(), levels=[ctr], colors='k')
    
    ax[i*3].format(title='',
                   ylabel='t = 13 hours',
                   xlabel='x-distance (m)')
    ax[i*3].set_xlim([0, 9000])
    ax[i*3].set_ylim([-200, 0])
    
    ax[i*3+1].format(title='',
                     ylabel='t = 54.5 hours',
                     xlabel='x-distance (m)')
    ax[i*3+1].set_xlim([0, 9000])
    ax[i*3+1].set_ylim([-200, 0])
    
    ax[i*3+2].format(title='',
                     ylabel='t = 75.5 hours',
                     xlabel='x-distance (m)')
    ax[i*3+2].set_xlim([0, 9000])
    ax[i*3+2].set_ylim([-200, 0])

ax[6].colorbar(m1, loc='b', width=0.1, ticks=0.1)
ax[7].colorbar(m2, loc='b', width=0.1, ticks=1)
ax[8].colorbar(m3, loc='b', width=0.1, ticks=5)


#%%
import proplot as pplt
import matplotlib.pyplot as plt

contours = np.array([25.775, 25.796, 25.82])
times = np.array([13, 54.5, 75.5]) * 3600

fig, ax = pplt.subplots(nrows=2, ncols=3, figsize=(11,6),
                        wspace=(0.1, 0.1), hspace=(0.3))

ax.format(abc=True, abcloc='l', abcstyle='(a)')

for i, (ctr, tim) in enumerate(zip(contours, times)):
    m1 = ax[i].contourf(tr.sel(time=tim).squeeze(), cmap=plt.cm.jet,
                          levels=np.linspace(25.75, 26.25, 41), extend='both')
    m2 = ax[i+3].contourf(np.log(dYsmth.sel(time=tim-300)), cmap=plt.cm.jet,
                          levels=np.linspace(-9, -2, 41), extend='both')
    
    ax[i].contour(tr.sel(time=tim).squeeze(), levels=[ctr], colors='k')
    
    ax[i].format(title='tracer ({} hr)'.format(tim/3600),
                   ylabel='depth (m)',
                   xlabel='x-distance (m)')
    ax[i].set_ylabel('depth (m)', fontsize=13)
    ax[i].set_xlim([0, 9000])
    ax[i].set_ylim([-200, 0])
    
    ax[i+3].format(title='local disp. rate ({} hr)'.format(tim/3600),
                     ylabel='depth (m)',
                     xlabel='x-distance (m)')
    ax[i+3].set_xlim([0, 9000])
    ax[i+3].set_ylim([-200, 0])

ax[2].colorbar(m1, loc='r', width=0.1, ticks=0.05)
ax[5].colorbar(m2, loc='r', width=0.1, ticks=1)
