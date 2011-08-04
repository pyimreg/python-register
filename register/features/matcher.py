""" A collection of feature matchers"""

import numpy as np
import scipy.ndimage as nd
import math

import matplotlib.pyplot as plt

def _getPointOffset(refChip, rawChip, padSize = 64):    

    diffX = rawChip.shape[1] - refChip.shape[1]
    diffY = rawChip.shape[0] - refChip.shape[0]

    refChip = refChip - np.mean(refChip)
    rawChip = rawChip - np.mean(rawChip)
    
    paddedRefChip = np.zeros([padSize,padSize])
    paddedRawChip = np.zeros([padSize,padSize])
    paddedRefChip[:refChip.shape[0],:refChip.shape[1]] = refChip 
    paddedRawChip[:rawChip.shape[0],:rawChip.shape[1]] = rawChip 
    
    rawFFTarray = np.fft.fft2(paddedRawChip)
    refFFTarray = np.fft.fft2(paddedRefChip)
    
    normCongProduct = (rawFFTarray*refFFTarray.conj())#/np.abs(rawFFTarray*refFFTarray.conj())

    ncOutputArray = np.fft.fftshift(np.fft.ifft2(normCongProduct))

    arg = np.argmax(np.abs(ncOutputArray))
    dims = ncOutputArray.shape
    idx = np.unravel_index(arg,dims) 
    
    return (idx[1] - (padSize/2) - diffX/2, idx[0] - (padSize/2) - diffY/2, np.abs(ncOutputArray[idx[0], idx[1]]))


def phaseCorrelationMatch(refdata, inputimage, chipsize=32, searchsize=64, threshold=0.0):
    plt.ion()
    matchedfeatures = {} 
    for id, point in refdata.features['points'].items():
        if point[0] > chipsize/2 and point[1] > chipsize/2 and point[0] < refdata.data.shape[0] - chipsize/2 and point[1] < refdata.data.shape[1] - chipsize/2:
            refChip = refdata.data[point[0]-chipsize/2:point[0]+chipsize/2,
                                   point[1]-chipsize/2:point[1]+chipsize/2] 
            inpChip = inputimage[point[0]-searchsize/2:point[0]+searchsize/2,
                                 point[1]-searchsize/2:point[1]+searchsize/2] 
            
            offsetX, offsetY, correlation = _getPointOffset(refChip, inpChip, padSize=searchsize)
            
            print "(%d, %d) -> %f" % (offsetX, offsetY, correlation)
            if correlation > threshold:
                matchedfeatures[id] = (point[0] + offsetY, point[1] + offsetX) 
    
    result = {}
    result['points'] = matchedfeatures
    return result
