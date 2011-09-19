""" A collection of feature matchers"""

import numpy as np
import scipy.ndimage as nd
import math
import sys

from register.features import region
from register.features import distance

import matplotlib.pyplot as plt

def _getPointOffset(refChip, rawChip, padSize = 64):    

    diffX = rawChip.shape[1] - refChip.shape[1]
    diffY = rawChip.shape[0] - refChip.shape[0]

    #refChip = refChip - np.mean(refChip)
    #rawChip = rawChip - np.mean(rawChip)
    
    #paddedRefChip = np.zeros([padSize,padSize])
    #paddedRawChip = np.zeros([padSize,padSize])
    #paddedRefChip[:refChip.shape[0],:refChip.shape[1]] = refChip 
    #paddedRawChip[:rawChip.shape[0],:rawChip.shape[1]] = rawChip 
    
    rawFFTarray = np.fft.fft2(rawChip)#, padSize)
    refFFTarray = np.fft.fft2(refChip)#, padSize)
    
    normCongProduct = (rawFFTarray*refFFTarray.conj())/np.abs(rawFFTarray*refFFTarray.conj())

    ncOutputArray = np.fft.fftshift(np.fft.ifft2(normCongProduct))

    arg = np.argmax(np.abs(ncOutputArray))
    dims = ncOutputArray.shape
    idx = np.unravel_index(arg,dims) 
    
    return (idx[1] - (padSize/2) - diffX/2, idx[0] - (padSize/2) - diffY/2, np.abs(ncOutputArray[idx[0], idx[1]]))


def phaseCorrelationMatch(refdata, inputimage, chipsize=32, searchsize=64, threshold=0.0):
    matchedfeatures = {} 
    for id, point in refdata.features['points'].items():
        if point[0] > chipsize/2 and point[1] > chipsize/2 and point[0] < refdata.data.shape[0] - chipsize/2 and point[1] < refdata.data.shape[1] - chipsize/2:
            refChip = refdata.data[point[0]-chipsize/2:point[0]+chipsize/2,
                                   point[1]-chipsize/2:point[1]+chipsize/2] 
            inpChip = inputimage[point[0]-searchsize/2:point[0]+searchsize/2,
                                 point[1]-searchsize/2:point[1]+searchsize/2] 
            
            offsetX, offsetY, correlation = _getPointOffset(refChip, inpChip, padSize=searchsize)
            
            if (correlation > threshold) and (offsetX >= 0) and (offsetY >= 0) and (offsetX < inputimage.shape[1]) and (offsetY < inputimage.shape[0]):
                matchedfeatures[id] = (point[0] + offsetY, point[1] + offsetX) 
    
    result = {}
    result['points'] = matchedfeatures
    return result

def regionCovarianceMatch(refdata, inputimage, chipsize=32, searchsize=64):
    matchedfeatures = {} 
    searchDist = (searchsize-chipsize)/2
    for id, point in refdata.features['points'].items():
        print id, point
        mindist = np.finfo(np.float).max
        mindistRC = None
        for r in range(point[0]-searchDist, point[0]+searchDist):
            for c in range(point[1]-searchDist, point[1]+searchDist):
                R = inputimage[r:r+chipsize, c:c+chipsize]
                C_R = region.covariance(R)
                C_F = refdata.features['regionCovariances'][id]
                d = distance.frobeniusNorm(C_F - C_R)
                if d < mindist:
                    mindist = d
                    mindistRC = (r,c)
        if not mindistRC is None:
            print mindist
            matchedfeatures[id] = (mindistRC[0] + chipsize/2, mindistRC[1] + chipsize/2)
    result = {}
    result['points'] = matchedfeatures
    return result
    
