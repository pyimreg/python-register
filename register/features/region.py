""" Helper functions for region covariance based feature matchers"""

import numpy as np
import scipy.ndimage as nd

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
        Pi = _integral(featureDescriptors[:,:,i])
        p_xy[i,:,:] = Pi
        for j in range(d):
            Pj = _integral(featureDescriptors[:,:,j])
            Qij = Pi*Pj
            Q_xy[i,j,:,:] = Qij
    return (p_xy, Q_xy)

_saved_work = {}

def flush_work():
    """ Flush all saved work for previously processed images from memory. 
    """
    _saved_work = {}
    
def covariance(image, upperleft, lowerright):
    """ Calculates the 7x7 covariance matrix for the region specified. 
    
        @param image: The input image (MxN)
        @return: The featureDescriptors (MxNx7) of type float 
    """
    global _saved_work
    if not _saved_work.has_key(image.__hash__()):
        featureDescriptors = _featureDescriptors(image)
        p, Q = _p_Q(featureDescriptors)
        _saved_work[image.__hash__()] = (p, Q, featureDescriptors)
    else:
        p, Q, featureDescriptors = _saved_work[image.__hash__()]
    QQQQ = Q[:,:,lowerright[1], lowerright[0]] + Q[:,:,upperleft[1], upperleft[0]] - Q[:,:,upperleft[1], lowerright[0]] - Q[:,:,lowerright[1], upperleft[0]]
    pppp = p[:,lowerright[1], lowerright[0]] + p[:,upperleft[1], upperleft[0]] - p[:,upperleft[1], lowerright[0]] - p[:,lowerright[1], upperleft[0]]
    n = (lowerright[1] - upperleft[1])*(lowerright[0] - upperleft[0])
    C = 1.0 / (n - 1.0)  * (QQQQ - 1.0 / n * pppp*pppp.T)
    return C
