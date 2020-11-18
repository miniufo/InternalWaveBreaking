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


trName = 'THETA'

path = 'I:/breakingIW/nonhydro/'

ds = xmitgcm.open_mdsdataset(data_dir=path+'output/',
                             grid_dir=path,
                             delta_t=2,
                             prefix=['Stat'])

ds2 = xr.open_dataset(path + 'KeffThInterp.nc').rename({'new':'Z'})
# ds2.coords['time'] = np.linspace(0, 300*(tcount-1), tcount).astype('int')

#%% Calculate N2

T = ds.THETA.where(ds.THETA!=0)

def cal_linear_insitu_density(THETA, SALT,
                           tRef  =20  , sRef=35, rhoRef=999.8,
                           tAlpha=2E-4, sBeta=7.4E-4):
        # cal. in-situ density using linear EOS
        RHO = rhoRef * (sBeta * (SALT-sRef) - tAlpha * (THETA - tRef)) + rhoRef
        
        RHO.rename('RHO')
        
        return RHO

density = cal_linear_insitu_density(ds2.THETA, 35, sRef=35, sBeta=0)

N2 = 9.81 * np.log(density).diff('Z')
N22 = 9.81 * (-2E-4*T/(1-2E-4*(T-20))).diff('Z')

N2smth = N2.rolling(Z=11, center=True, min_periods=1).mean() * 1e5

# N2smth = N2smth.rename({'new':'Z'})


#%% plotting
import proplot as pplt

array = [
    [1, 1, 3],
    [2, 2, 4]
]

pplt.rc.update({'figure.facecolor': 'white'})


fig, ax = pplt.subplots(array, figsize=(9,7), sharex=0, sharey=3)

fontsize = 13

ax.format(abc=True, abcloc='l', abcstyle='(a)')

lastIdx = -4

m1 = ax[0].contourf(T[0].squeeze(), cmap=plt.cm.jet, levels=np.linspace(25.75, 26.25, 21))
ax[0].colorbar(m1, loc='bottom', width=0.1, ticks=0.05, extend='both')
ax[0].set_xlabel('', fontsize=fontsize-2)
ax[0].set_ylabel('depth (m)', fontsize=fontsize-2)
ax[0].set_title('potential temperature (t=0 days)', fontsize=fontsize)
ax[0].set_xlim([0, 9000])
ax[0].set_ylim([-200, 0])

ax[1].contourf(T[lastIdx].squeeze(), cmap=plt.cm.jet, levels=np.linspace(25.75, 26.25, 21))
ax[1].set_xlabel('x-distance (m)', fontsize=12)
ax[1].set_ylabel('depth (m)', fontsize=fontsize-2)
ax[1].set_title('potential temperature (t=5 days)', fontsize=fontsize)
ax[1].set_xlim([0, 9000])
ax[1].set_ylim([-200, 0])

m1 = ax[2].plot(N2smth[ 0], N2smth.Z, lw=2, label=['day 0'])
ax[2].set_ylabel('depth (m)', fontsize=fontsize-2)
m2 = ax[2].plot(N2smth[-1], N2smth.Z, lw=2, label=['day 5'])
ax[2].legend([m1, m2], loc='lr')
ax[2].set_xlabel('$N^2$ ($10^{-5}$ $s^{-1})$', fontsize=fontsize-2)
ax[2].set_ylabel('depth (m)', fontsize=fontsize-2)
ax[2].set_title('$N^2$ after sorting', fontsize=fontsize)
ax[2].set_xlim([0, 2])

m1 = ax[3].plot(ds2.THETA[ 0], ds2.Z, lw=2, label=['day 0'])
ax[3].set_ylabel('depth (m)', fontsize=fontsize-2)
m2 = ax[3].plot(ds2.THETA[lastIdx], ds2.Z, lw=2, label=['day 5'])
ax[3].legend([m1, m2], loc='lr')
ax[3].set_xlabel('temperature (degree)', fontsize=fontsize-2)
ax[3].set_ylabel('depth (m)', fontsize=fontsize-2)
ax[3].set_title('profile after sorting', fontsize=fontsize)
ax[3].set_xlim([25.75, 26.25])

ax.format(ylabel='depth (m)')

