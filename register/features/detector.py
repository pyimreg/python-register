""" A collection of feature detectors"""

import numpy as np
import scipy.ndimage as nd
import math

from register.features.haar2d import haar2d

__debug=False


def _debug(something):
    global __debug
    if __debug:
        print something


# Constants
HaarDetector = 0

def detect(image, detectorType=HaarDetector, options=None, debug=False):
    global __debug
    global __plt
    __debug = debug
    if detectorType == HaarDetector:
        return _detectHaarFeatures(image, options)
    else:  # default detector
        return _detectHaarFeatures(image, options)


def _haarDefaultOptions(image):
    options = {}
    options['levels'] = 5         # number of wavelet levels
    options['threshold'] = 0.2    # threshold between 0.0 and 1.0 to filter out weak features (0.0 includes all features)
    options['locality'] = 5       # minimum (approx) distance between two features  
    return options

def _detectHaarFeatures(image, options={}):
    if options is None:
        options = _haarDefaultOptions(image)
    levels = options.get('levels')   
    maxpoints = options.get('maxpoints')
    threshold = options.get('threshold')
    locality = options.get('locality')
    
    haarData = haar2d(image, levels)

    avgRows = haarData.shape[0] / 2 ** levels
    avgCols = haarData.shape[1] / 2 ** levels
    
    SalientPoints = {}
    
    siloH = np.zeros([haarData.shape[0]/2, haarData.shape[1]/2, levels])
    siloD = np.zeros([haarData.shape[0]/2, haarData.shape[1]/2, levels])
    siloV = np.zeros([haarData.shape[0]/2, haarData.shape[1]/2, levels])
    
    # Build the saliency silos
    for i in range(levels):
        level = i + 1
        halfRows = haarData.shape[0] / 2 ** level
        halfCols = haarData.shape[1] / 2 ** level
        siloH[:,:,i] = nd.zoom(haarData[:halfRows, halfCols:halfCols*2], 2**(level-1)) 
        siloD[:,:,i] = nd.zoom(haarData[halfRows:halfRows*2, halfCols:halfCols*2], 2**(level-1)) 
        siloV[:,:,i] = nd.zoom(haarData[halfRows:halfRows*2, :halfCols], 2**(level-1)) 
    
    # Calculate saliency heat-map
    saliencyMap = np.max(np.array([
                                np.sum(np.abs(siloH), axis=2), 
                                np.sum(np.abs(siloD), axis=2),
                                np.sum(np.abs(siloV), axis=2)
                                ]), axis=0)
                               
    # Determine global maximum and saliency threshold
    maximum = np.max(saliencyMap)
    sthreshold = threshold * maximum
    
    # Extract features by finding local maxima
    rows = haarData.shape[0] / 2
    cols = haarData.shape[1] / 2
    features = {}
    id = 0
    for row in range(locality,rows-locality):
        for col in range(locality,cols-locality):
            saliency = saliencyMap[row,col]
            if saliency > sthreshold:
                if  saliency >= np.max(saliencyMap[row-locality:row+locality, col-locality:col+locality]):
                    features[id] = (row*2,col*2)
                    id += 1

    result = {}
    result['points'] = features
    return result
    



