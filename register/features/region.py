""" Helper functions for region covariance based feature matchers"""

import numpy as np
import scipy.ndimage as nd
import hashlib

def _integral(image):
    """ Calculates the integral image of the image provided. 
    
        @param image: The input image (MxN)
        @return: The integral image (MxN) of type float 
    """
    sum = 0.0
    result = np.empty_like(image).astype(np.float)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            sum += image[row,col]
            result[row, col] = sum
    return result
            
        

def _featureDescriptors(image):
    """ Calculates the 7 element feature desciptor vector for each pixel of the image provided. 
    
        @param image: The input image (MxN)
        @return: The featureDescriptors (MxNx7) of type float 
    """
    dIdx = nd.filters.convolve(image, np.array([[-1, 0, 1]]))
    d2Idx2 = nd.filters.convolve(image, np.array([[-1, 2, -1]]))
    dIdy = nd.filters.convolve(image, np.array([[-1, 0, 1]]).T)
    d2Idy2 = nd.filters.convolve(image, np.array([[-1, 2, -1]]).T)
    row = np.empty(image.shape)
    col = np.empty(image.shape)
    row[:, :] = np.arange(image.shape[0])
    row = row.T
    col[:, :] = np.arange(image.shape[1])
    result = np.empty([image.shape[0], image.shape[1], 7])
    result[:,:, 0] = row
    result[:,:, 1] = col
    result[:,:, 2] = image
    result[:,:, 3] = dIdx
    result[:,:, 4] = dIdy
    result[:,:, 5] = d2Idx2
    result[:,:, 6] = d2Idy2
    return result


def _p_Q(featureDescriptors):
    """ Calculates the p and Q matrices using the featureDescriptors.
    
        Derivation in: Region Covariance - A Fast Descriptor for Detection
        and Classification, Oncel Tuzel, Fatih Porikli, Peter Meer
    
        @param image: The featureDescriptors (MxNx7)
        @return: The p and Q matices 
    """
    d = featureDescriptors.shape[2]
    x = featureDescriptors.shape[1]
    y = featureDescriptors.shape[0]
    p_xy = np.empty([d,x,y])
    Q_xy = np.empty([d,d,x,y])
    for i in range(d):
        p_xy[i,:,:] = _integral(featureDescriptors[:,:,i])
        for j in range(d):
            Q_xy[i,j,:,:] = _integral(featureDescriptors[:,:,i]*featureDescriptors[:,:,j])
    return (p_xy, Q_xy)

_saved_work = {}


def covariance(image, upperleft=(0,0), lowerright=None):
    """ Calculates the 7x7 covariance matrix for the region specified. (lowerright is excluded rom the region). 
    
        @param image: The input image (MxN)
        @param upperleft: The upper left point (row,col)
        @param lowerright: The lower right point (row,col)
        @return: The covariance matrix (7x7) of the region specified. 
    """
    if lowerright is None:
        lowerright = image.shape
    region = image[upperleft[0]:lowerright[0], upperleft[1]:lowerright[1]]
    regionfeatures = _featureDescriptors(region)
    d = regionfeatures.shape[2]
    n = np.prod(region.shape)
    C = np.cov(regionfeatures.reshape([n,d]))            
    return C

def flush_work():
    """ Flush all saved work for previously processed images from memory. 
    """
    _saved_work = {}
    
def imagehash(image):
    """ Determine the image hash. 
    """
    return hashlib.md5(image.dumps()).hexdigest()

def covariance_(image, upperleft, lowerright, imagekey=None):
    """ Calculates the 7x7 covariance matrix for the region specified. 
    
        Derivation in: Region Covariance - A Fast Descriptor for Detection
        and Classification, Oncel Tuzel, Fatih Porikli, Peter Meer
    
        @param image: The input image (MxN)
        @param upperleft: The upper left point (row,col)
        @param lowerright: The lower right point (row,col)
        @param imagekey: The input image's hash. Faster if this is calculated only once and passed in repeatedly
        @return: The covariance matrix (7x7) of the region specified. 
    """
    global _saved_work
    if imagekey is None:
        imagekey=imagehash(image)
    if not _saved_work.has_key(imagekey):
        featureDescriptors = _featureDescriptors(image)
        p, Q = _p_Q(featureDescriptors)
        _saved_work[imagekey] = (p, Q, featureDescriptors)
    else:
        p, Q, featureDescriptors = _saved_work[imagekey]
    QQQQ = Q[:,:,lowerright[1], lowerright[0]] + Q[:,:,upperleft[1], upperleft[0]] - Q[:,:,upperleft[1], lowerright[0]] - Q[:,:,lowerright[1], upperleft[0]]
    pppp = p[:,lowerright[1], lowerright[0]] + p[:,upperleft[1], upperleft[0]] - p[:,upperleft[1], lowerright[0]] - p[:,lowerright[1], upperleft[0]]
    n = (lowerright[1] - upperleft[1])*(lowerright[0] - upperleft[0])
    C = 1.0 / (n - 1.0) * (QQQQ - 1.0 / n * pppp*pppp.T)
    return C
