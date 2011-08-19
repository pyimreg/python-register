import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.io import netcdf

import advect

VMIN=980
VMAX=1050
CMAP='jet'

#==============================================================================

params = {'axes.labelsize': 10,
          'axes.titlesize': 10,
          'figure.titlesize': 14,
          'font.size': 10,
          'font.weight':'normal',
          'text.fontsize': 10,
          'axes.fontsize': 10,
          'legend.fontsize': 11,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'figure.figsize': (12,6),
          'figure.facecolor': 'w',
          }

plt.rcParams.update(params)

#==============================================================================

def sampleGrid(data, mapbounds, subbounds):

    m = Basemap(
        projection='cyl',
        llcrnrlat=mapbounds[2],
        urcrnrlat=mapbounds[3],
        llcrnrlon=mapbounds[0],
        urcrnrlon=mapbounds[1],
        resolution='i'
        )
    
    coords = m.makegrid( data.shape[1], data.shape[0])
    
    mask = np.zeros(data.shape, dtype=np.bool)
    
    mask[ (coords[0] > subbounds[0]) * 
          (coords[0] < subbounds[1]) * 
          (coords[1] > subbounds[2]) * 
          (coords[1] < subbounds[3]) ] = True
    
    (r, c) = np.nonzero(mask)
    
    shape = (max(r) - min(r) + 1, max(c) - min(c) +1)
    
    return np.reshape(data[mask], shape)

def mapGrid(ax, grid, mapbounds, vmin=VMIN, vmax=VMAX, cb=True, overlay=True):
    
    cstd = ax.imshow(
        grid, 
        origin='lower',
        extent=mapbounds,
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
        alpha=0.9,
        cmap=CMAP
        )
    
    if cb:
        cb = plt.colorbar(cstd, shrink=0.5)
        cb.ax.set_xlabel('hPa')
    
    isobars = ax.contour(
        grid, 
        xrange(vmin,vmax,2),
        origin='lower',
        extent=mapbounds,
        colors='black',
        alpha=0.5
        )
    
    m = Basemap(
          projection='cyl',
          llcrnrlat=mapbounds[2],
          urcrnrlat=mapbounds[3],
          llcrnrlon=mapbounds[0],
          urcrnrlon=mapbounds[1],
          resolution='c')
    
    if overlay:
        m.drawcoastlines()
   
    m.drawparallels(np.arange(mapbounds[2], mapbounds[3], 5.), labels=[1,0,0,1], ax=ax)
    m.drawmeridians(np.arange(mapbounds[0], mapbounds[1], 5.), labels=[1,0,0,1], ax=ax)
    
    ax.grid()

dataset = '/work/archive/{}/{}'

netcdf = sp.io.netcdf.netcdf_file(
    dataset.format('CYCLONE_YASI', 'QNHObs.nc'), 
    'r'
    )

grids = netcdf.variables['QNHObs_SFC']


mapbounds = [109.5, 163.5, -45.5, -8.5]
subbounds = [148.5, 163.5, -20.0, -8.5]

A = grids[55]
subA = sampleGrid(A, mapbounds, subbounds) 

B = grids[35]
subB = sampleGrid(B, mapbounds, subbounds) 

C = grids[45]
subC = sampleGrid(C, mapbounds, subbounds) 


advections = advect.warp(
    subA,
    subB,
    samples=40,
    multiStage=False
    )

plt.figure()
mapGrid(plt.subplot(1,2,1), A, mapbounds)
plt.title('(A) (t=0)')
mapGrid(plt.subplot(1,2,2), B, mapbounds)
plt.title('(B) (t=-20)')
plt.suptitle('Tropical cyclone YASI', weight='bold')

#plt.text(
#    mapbounds[0], 
#    mapbounds[2]+0.1,
#    "Equidistant Cylindrical Projection [{0}, {1}, {2}, {3}]".format(
#    *mapbounds
#    )
#    )

midpoint = int(len(advections)/2.0)


plt.figure()
mapGrid(plt.subplot(2,2,1), subA, subbounds)
plt.title('(A) (t=0)')
mapGrid(plt.subplot(2,2,2), subB, subbounds)
plt.title('(B) (t=-20)')
mapGrid(plt.subplot(2,2,3), advections[midpoint].forward, subbounds)
plt.title('A advected to B (50%)')
mapGrid(plt.subplot(2,2,4), advections[midpoint].inverse, subbounds)
plt.title('B advected to A (50%)')

weightedSum = 0.5 * (subA + subC)
weightedSumError = weightedSum-subC
advectedWeightedSum = 0.5 * (advections[midpoint].forward + advections[midpoint].inverse)
advectedSumError =  advectedWeightedSum-subC

plt.figure()
mapGrid(plt.subplot(2,3,1), weightedSum, subbounds, overlay=False)
plt.title('(linear) weighted sum')
mapGrid(plt.subplot(2,3,2), advectedWeightedSum, subbounds, overlay=False)
plt.title('(advected) weighted sum')
mapGrid(plt.subplot(2,3,3), subC, subbounds, overlay=False)
plt.title('Observation')
mapGrid(plt.subplot(2,3,4), weightedSum-subC, subbounds, vmin=-10, vmax=10, overlay=False)
plt.title('(linear) absolute error')
mapGrid(plt.subplot(2,3,5), advectedWeightedSum-subC, subbounds, vmin=-10, vmax=10, overlay=False)
plt.title('(advected) absolute error')



plt.ion()
plt.figure()

ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

import os
import sys

files = []
for index, advection in enumerate(advections):
    
    scale = advection.scale
    advectedWeightedSum = (1.0-scale)*advection.forward + scale*advection.inverse
    weightedSum = (1.0-scale)*subA + scale*subB
    mapGrid(ax1, advectedWeightedSum, subbounds, cb=False, overlay=False)
    plt.draw()
    mapGrid(ax2, weightedSum, subbounds, cb=False, overlay=False)
    plt.draw()

    fname = '/tmp/_tmp%03d.png' % index
    plt.savefig(fname)
   
    ax1.cla()
    ax2.cla()
    
   
plt.ioff()

plt.show()








