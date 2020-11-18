# -*- coding: utf-8 -*-
"""
Created on 2020.02.28

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

tcount = len(dset.time)

dsKeff = xr.open_dataset(path + 'keff.nc')

dsKeff.coords['time'] = np.linspace(0, 300*(tcount-1), tcount).astype('int')
dsKeff = dsKeff.rename({'new':'Z'})


#%% all
import numpy as np
from GeoApps.GridUtils import add_MITgcm_missing_metrics
from GeoApps.ContourMethods import ContourAnalysis


def get_group(groups, name):
    """
    Get DataArray or Dataset corresponding to a particular group label.
    """
    gr = groups.groups
    
    if name not in gr:
        return None

    indices = gr.get(name)
    return groups._obj.isel(**{groups._group_dim: indices}).values


dt = 300
tlen = 100

reM = []
reD = []
reC = []

for l in range(1, tcount, tlen):
    tstr = l
    tend = l + tlen
    
    if tend >= tcount:
        tend = tcount-2
    
    rng  = dict(time=slice(tstr*dt, tend*dt))
    
    print('%d  %d' % (tstr, tend))
    
    # cal_Yeq
    ds = dset.loc[rng]
    ds, grid = add_MITgcm_missing_metrics(ds)
    
    mask   = ds['TRAC01'] != 0
    tracer = ds['TRAC01'].where(mask)
    
    cm = ContourAnalysis(ds, tracer,
                         dims={'X':'XC','Z':'Z'},
                         dimEq={'Z':'Z'}, grid=grid,
                         increase=True, lt=False)
    
    maskInvariant = tracer[0]
    maskInvariant += 1 - maskInvariant
    
    ctr   = dsKeff.TRAC01
    area  = dsKeff.intArea
    table = cm.cal_area_eqCoord_table(maskInvariant)
    Zeq   = dsKeff.zEq
    
    # read_flt
    from utils.IOUtils import read_flt_bcolz
    
    fset = read_flt_bcolz(path + '.bcolz',
                    cond="(time>="+str(tstr*dt)+") & (time<="+str(tend*dt)+")",
                    # cond="(npart>10000000) & (npart<=10001000) & (time>0)",
                    fields=['npart', 'time', 'x', 'z', 's'],
                    out='xarray').dropna('npart')
    
    # interpolate
    # xpos = fset.x.where(fset.s!=0).dropna('time',
    #                                       how='all').dropna('npart').loc[rng]
    # zpos = fset.z.where(fset.s!=0).dropna('time',
    #                                       how='all').dropna('npart').loc[rng]
    xpos = fset.x.loc[rng]
    zpos = fset.z.loc[rng]
    time = fset.time.loc[rng]
    
    var1 = tracer.squeeze().loc[rng].load()
    
    if len(time) > len(var1.time):
        print(str(len(time)) + ' != ' + str(len(var1.time)))
        time = time[:len(var1.time)]
        xpos = xpos[:, :len(var1.time)]
        zpos = zpos[:, :len(var1.time)]
        
    xpos, time = xr.broadcast(xpos, time)
    
    trajT = var1.interp(coords={'XC':xpos, 'Z':zpos, 'time':time}) \
                .reset_coords(drop=True) \
                .dropna('npart')
    trajY = cm.interp_to_coords(trajT,
                                ctr.loc[rng], Zeq.loc[rng],
                                interpDim='Z') \
              .transpose('npart','time') \
              .reset_coords(drop=True)
    # trajY = xr.DataArray(table.lookup_coordinates(trajA),
    #                      dims=trajA.dims, coords=trajA.coords)
    # del trajA
    
    # dispersion
    for ll in range(tstr, tend):
        timNp1 = dt * (ll+1)
        timN   = dt *  ll
        
        contour =   ctr.sel(time=timN).load()
        trajVal = trajT.sel(time=timN)
        
        Ynp1    = trajY.sel(time=timNp1)
        Yn      = trajY.sel(time=timN)
        
        # group npart into contour bins
        reverse = xr.DataArray(trajVal.npart, dims='contours',
                               coords={'contours':trajVal.values}).sortby('contours')
        
        bins = contour.values[::-1]
        groups = reverse.groupby_bins('contours', bins, labels=bins[:-1])
        
        # if len(groups._group_indices) != 199:
        #     print(len(groups._group_indices))
        #     raise Exception('okok')
        
        tmpD = []
        tmpM = []
        tmpC = []
        
        for name in bins[::-1]:
            ids = get_group(groups, name)
            
            if ids is None:
                tmpD.append(xr.DataArray(np.nan))
                tmpM.append(xr.DataArray(np.nan))
            else:
                disp = (Ynp1.sel(npart=ids) - Yn.sel(npart=ids))
                disM = disp.mean('npart')
                disA = disp - disM
                diff = (disA**2).mean('npart')
                tmpD.append(diff / (dt * 2.0))
                tmpM.append(disM)
            tmpC.append(xr.DataArray(name))
        
        reD.append(xr.concat(tmpD, dim='contour'))
        reM.append(xr.concat(tmpM, dim='contour'))
        reC.append(xr.concat(tmpC, dim='contour'))

#%%
diffu = xr.concat(reD, dim='time').rename('diffu')
drift = xr.concat(reM, dim='time').rename('drift')
ctrDA = xr.concat(reC, dim='time').rename('ctrDA')

length = tcount-3

diffu.coords['time'   ] = dset.time[0:length]
diffu.coords['contour'] = dsKeff['Z'].values
drift.coords['time'   ] = dset.time[0:length]
drift.coords['contour'] = dsKeff['Z'].values
ctrDA.coords['time'   ] = dset.time[0:length]
ctrDA.coords['contour'] = dsKeff['Z'].values

diffu = diffu.rename({'contour':'Z'})
drift = drift.rename({'contour':'Z'})
ctrDA = ctrDA.rename({'contour':'Z'})

diffu.coords['time'] = np.linspace(0, 300*(length-1), length)
drift.coords['time'] = np.linspace(0, 300*(length-1), length)
ctrDA.coords['time'] = np.linspace(0, 300*(length-1), length)


xr.merge([diffu, drift, ctrDA]).to_netcdf(path + 'LagDiff3.nc')

#%%
# preZ = np.linspace(0, -200, 101)

# diffuInterp = cm.interp_to_coords(preZ, zEq[:length].squeeze(), diffu)
# driftInterp = cm.interp_to_coords(preZ, zEq[:length].squeeze(), drift)
# ctrDAInterp = cm.interp_to_coords(preZ, zEq[:length].squeeze(), ctrDA)

#%%
import matplotlib.pyplot as plt

dsLag1 = xr.open_dataset(path + 'LagDiff3.nc')
# dsLag1 = dsLag1.rename({'contour':'Z'})
dsLag2 = xr.open_dataset(path + 'LagDiff2.nc')

#%%
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,8))

# .rolling(time=1, center=True, min_periods=1).mean()

diff1 = dsLag1.diffu * 5e3
diff1.plot(ax=ax[0], y='Z', cmap='jet', vmin=0, vmax=12)

diff2 = dsLag2.diffu * 5e3
diff2.plot(ax=ax[1], y='Z', cmap='jet', vmin=0, vmax=12)



