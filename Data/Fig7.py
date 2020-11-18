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

tcount = len(dset.time)

dsKeff = xr.open_dataset(path + 'keff.nc')

dsKeff.coords['time'] = np.linspace(0, 300*(tcount-1), tcount).astype('int')
dsKeff = dsKeff.rename({'new':'Z'})

#%%
import numpy as np
from GeoApps.GridUtils import add_MITgcm_missing_metrics
from GeoApps.ContourMethods import ContourAnalysis
from xhistogram.xarray import histogram


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

histSum = []
histCnt = []

binX = np.linspace(0, 9000, 501)
binZ = np.linspace(-200, 0, 41)

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
    
    diffY = trajY.diff('time') ** 2 / 300
              
    # remove nan according to trajY
    diffY, xpos = xr.align(diffY, xpos)
    diffY, zpos = xr.align(diffY, zpos)
    
    
    hsum = histogram(zpos, xpos, dim=['npart'], weights=np.abs(diffY), 
                     bins=[binZ, binX])
    hcnt = histogram(zpos, xpos, dim=['npart'],
                     bins=[binZ, binX])
    
    histSum.append(hsum)
    histCnt.append(hcnt)


# hstSum = xr.merge(histSum)
# hstCnt = xr.merge(histCnt)


#%% write files
for data in histSum:
    for tim in data:
        tt = int(tim.time.values/300)
        print(tt)
        tim.expand_dims({'time':1}).to_netcdf("d:/tmp/testSum_{:0>4d}.nc".format(tt),
                                              mode='w',
                                              engine='netcdf4',
                      format='NETCDF4') # 'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT',

for data in histCnt:
    for tim in data:
        tt = int(tim.time.values/300)
        print(tt)
        tim.expand_dims({'time':1}).to_netcdf("d:/tmp/testCnt_{:0>4d}.nc".format(tt),
                                              mode='w',
                                              engine='netcdf4',
                      format='NETCDF4') # 'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT',

#%% combine
dsSum = xr.open_mfdataset('d:/tmp/testSum_*.nc')
dsCnt = xr.open_mfdataset('d:/tmp/testCnt_*.nc')

xr.merge([dsSum.rename({'histogram_z_x':'sum'}),
          dsCnt.rename({'histogram_z_x':'count'})]).to_netcdf('d:/tmp/testDisp.nc')

#%%
bset = xr.open_dataset('d:/tmp/testDisp.nc')

# smooth = bset['sum'  ].rolling(time=3, center=True, min_periods=1).mean()
# count  = bset['count'].rolling(time=3, center=True, min_periods=1).mean()
#%%
delY = bset['sum']/bset['count']

dYsmth = delY.rolling(time=5,
                      center=True,
                      min_periods=1).mean().rolling(x_bin=3,center=True,
                                                    min_periods=1).mean()

#%%
ani=animate(dYsmth, cmap='jet', vmax=2)
ani2=animate(dset.TRAC01.where(dset.TRAC01!=0).squeeze(), cmap='jet',vmin=25.67)

#%%


