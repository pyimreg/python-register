""" Helper functions for region covariance based feature matchers"""

import numpy as np
import scipy.ndimage as nd

def integral(image):
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
            
        

def featureDescriptors(image):
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


def p_Q(featureDescriptors):
    d = featureDescriptors.shape[2]
    x = featureDescriptors.shape[1]
    y = featureDescriptors.shape[0]
    p_xy = np.empty([d,x,y])
    Q_xy = np.empty([d,d,x,y])
    for i in range(d):
        Pi = integral(featureDescriptors[:,:,i])
        p_xy[i,:,:] = Pi
        for j in range(d):
            Pj = integral(featureDescriptors[:,:,j])
            Qij = Pi*Pj
            Q_xy[i,j,:,:] = Qij
    return (p_xy, Q_xy)
    
            
