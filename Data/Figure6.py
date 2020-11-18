# -*- coding: utf-8 -*-
"""
Created on 2020.07.08

@author: MiniUFO
"""
#%% read_fields
import xarray as xr
import numpy as np

path = 'I:/breakingIW/nonHydro/'

bset1 = xr.open_dataset(path + 'diags/binning/binLagTh_dec_gt2.nc')
bset2 = xr.open_dataset(path + 'diags/binning/binLagThDq_dec_gt2.nc')
bset3 = xr.open_dataset(path + 'diags/nKeffXZ.nc',
                        chunks={'time':1, 'Z':100, 'XC':1440})
bset4 = xr.open_dataset(path + 'diags/Knum.nc', decode_times=False)

# dqPM = bset2.sumP.sum('time') / bset2.cntP.sum('time')
# dqNM = bset2.sumN.sum('time') / bset2.cntN.sum('time')

dYDM = bset1.sumD.sum('time') / bset1.cntD.sum('time') / bset4.Knum.mean('time')
dqDM = bset2.sumD.sum('time') / bset2.cntD.sum('time') * 1e9

dYDM2 = (bset1.sumD / bset4.Knum.rolling(time=11, center=True,
                                         min_periods=1).mean() / bset1.cntD).mean('time')
dqDM2 = (bset2.sumD / bset2.cntD).mean('time') * 1e5

keff = bset3.nkeff.mean('time').load()



#%%
import proplot as pplt
import matplotlib.pyplot as plt


fig, ax = pplt.subplots(nrows=1, ncols=3, figsize=(12,5), hspace=(0.1))

ax.format(abc=True, abcloc='l', abcstyle='(a)')

m1 = ax[0].contourf((keff-1), cmap=plt.cm.jet,
                    values=np.linspace(0, 13, 27), extend='both')
m2 = ax[1].contourf(dYDM.squeeze(), cmap=plt.cm.jet,
                    values=np.linspace(0, 13, 27), extend='both')
m3 = ax[2].contourf(dqDM.squeeze(), cmap=plt.cm.jet,
                    values=np.linspace(0, 9, 19), extend='both')

fontsize = 13

ax[0].set_title('time-mean $K_{WD}$', fontsize=fontsize)
ax[0].set_xlim([0, 8960])
ax[0].set_ylim([-200, 0])
ax[0].set_xlabel('x-distance (m)', fontsize=fontsize-2)
ax[0].set_ylabel('depth (m)', fontsize=fontsize-2)
ax[0].colorbar(m1, loc='b', width=0.15, ticks=1)

ax[1].set_title('time-mean $(\\Delta Y)^2/\\Delta t/2$', fontsize=fontsize)
ax[1].set_xlim([0, 8960])
ax[1].set_ylim([-200, 0])
ax[1].set_xlabel('x-distance (m)', fontsize=fontsize-2)
ax[1].set_ylabel('depth (m)', fontsize=fontsize-2)
ax[1].colorbar(m2, loc='b', width=0.15, ticks=1)

ax[2].set_title('time-mean $(\\Delta \\rho)^2/\\Delta t/2$', fontsize=fontsize)
ax[2].set_xlim([0, 8960])
ax[2].set_ylim([-200, 0])
ax[2].set_xlabel('x-distance (m)', fontsize=fontsize-2)
ax[2].set_ylabel('depth (m)', fontsize=fontsize-2)
ax[2].colorbar(m3, loc='b', width=0.15, ticks=1)


#%%
import proplot as pplt
import matplotlib.pyplot as plt

pplt.rc.update({'figure.facecolor': 'white'})


fig, ax = pplt.subplots(nrows=1, ncols=3, figsize=(9,4), hspace=(0.1))

ax.format(abc=True, abcloc='l', abcstyle='(a)')

m1 = ax[0].contourf(np.log(keff*2e-5*10), cmap=plt.cm.jet,
                    values=np.linspace(-10, -5, 31), extend='both')
m2 = ax[1].contourf(np.log(dYDM), cmap=plt.cm.jet,
                    levels=np.linspace(-10, -5, 31), extend='both')
m3 = ax[2].contourf(np.log(dqDM), cmap=plt.cm.jet, 
                    levels=np.linspace(-11, -6, 31), extend='both')

fontsize = 13

ax[0].set_title('time-mean $K_{WD}$', fontsize=fontsize)
ax[0].set_xlim([0, 8990])
ax[0].set_ylim([-200, 0])
ax[0].set_xlabel('x-distance (m)', fontsize=fontsize-2)
ax[0].set_ylabel('depth (m)', fontsize=fontsize-2)
ax[0].colorbar(m1, loc='b', width=0.15, ticks=1)

ax[1].set_title('time-mean $(\\Delta Y)^2/\\Delta t$', fontsize=fontsize)
ax[1].set_xlim([0, 8990])
ax[1].set_ylim([-200, 0])
ax[1].set_xlabel('x-distance (m)', fontsize=fontsize-2)
ax[1].set_ylabel('depth (m)', fontsize=fontsize-2)
ax[1].colorbar(m2, loc='b', width=0.15, ticks=1)

ax[2].set_title('time-mean $(\\Delta \\rho)^2/\\Delta t$', fontsize=fontsize)
ax[2].set_xlim([0, 8990])
ax[2].set_ylim([-200, 0])
ax[2].set_xlabel('x-distance (m)', fontsize=fontsize-2)
ax[2].set_ylabel('depth (m)', fontsize=fontsize-2)
ax[2].colorbar(m3, loc='b', width=0.15, ticks=1)
