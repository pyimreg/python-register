import numpy as np
import hashlib
import sys
import os


""" See http://en.wikipedia.org/wiki/Structural_similarity.

    Also: Region Covariance: A fast descriptor for detection and classification. 
"""


def ssim_direct(image, reference, bbox=None):
    if bbox is None:
        bbox = [0, 0, image.shape[1], image.shape[0]]
    img = image[bbox[1]:bbox[3], bbox[0]:bbox[2]].astype(np.uint8)
    ref = reference[bbox[1]:bbox[3], bbox[0]:bbox[2]].astype(np.uint8)
    
    meanI = np.mean(img.flatten())
    meanR = np.mean(ref.flatten())
    varI = np.cov(img.flatten(), img.flatten())[0,1]
    varR = np.cov(ref.flatten(), ref.flatten())[0,1]
    covIR = np.cov(img.flatten(), ref.flatten())[0,1]
    L = (2.0**8)-1.0
    k1 = 0.01
    k2 = 0.03
    c1 = (k1*L)**2.0
    c2 = (k2*L)**2.0
    SSIM = ((2.0*meanI*meanR + c1) * (2.0*covIR + c2)) / ((meanI**2.0 + meanR**2.0 + c1) * (varI + varR + c2))
    return SSIM

_saved_work = {}

def flush_work():
    """ Flush all saved work for previously processed images from memory. 
    """
    global _saved_work
    _saved_work = {}

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

def imageshash(image, reference):
    """ Determine the image pair's hash. 
    """
    return hashlib.md5(image.dumps()+reference.dumps()).hexdigest()

def _p_Q(stack):
    """ Calculates the p and Q matrices using the image stack.
    
        Derivation in: Region Covariance - A Fast Descriptor for Detection
        and Classification, Oncel Tuzel, Fatih Porikli, Peter Meer
    
        @param stack: The image stack (MxNx2)
        @return: The p and Q matices 
    """
    d = stack.shape[2]
    x = stack.shape[1]
    y = stack.shape[0]
    p_xy = np.empty([d,x,y])
    Q_xy = np.empty([d,d,x,y])
    for i in range(d):
        p_xy[i,:,:] = _integral(stack[:,:,i])
        for j in range(d):
            Q_xy[i,j,:,:] = _integral(stack[:,:,i]*stack[:,:,j])
    return (p_xy, Q_xy)


def ssim_fast(image, reference, bbox=None, imagekey=None):
    if bbox is None:
        bbox = [0, 0, image.shape[1], image.shape[0]]
    #assert len(bbox) == 4
    upperleft = [0,0]
    lowerright = [0,0]
    upperleft[0] = bbox[1]
    upperleft[1] = bbox[0]
    lowerright[0] = bbox[3] - 1
    lowerright[1] = bbox[2] - 1
    if imagekey is None:
        imagekey=imageshash(image,reference)
    if not _saved_work.has_key(imagekey):
        stack = np.dstack([image, reference])
        p, Q = _p_Q(stack)
        _saved_work[imagekey] = {'p':p, 'Q':Q}
    else:
        p = _saved_work[imagekey]['p']
        Q = _saved_work[imagekey]['Q']
    QQQQ = Q[:,:,lowerright[1], lowerright[0]] + Q[:,:,upperleft[1], upperleft[0]] - Q[:,:,upperleft[1], lowerright[0]] - Q[:,:,lowerright[1], upperleft[0]]
    pppp = p[:,lowerright[1], lowerright[0]] + p[:,upperleft[1], upperleft[0]] - p[:,upperleft[1], lowerright[0]] - p[:,lowerright[1], upperleft[0]]
    n = (lowerright[1] - upperleft[1])*(lowerright[0] - upperleft[0])
    C = 1.0 / (n - 1.0) * (QQQQ - 1.0 / n * pppp*pppp.T)
    meanI = 1.0 / n * pppp[0]
    meanR = 1.0 / n * pppp[1]
    varI = C[0,0]
    varR = C[1,1]
    covIR = C[0,1]
    L = (2.0**8)-1.0
    k1 = 0.01
    k2 = 0.03
    c1 = (k1*L)**2.0
    c2 = (k2*L)**2.0
    SSIM = ((2.0*meanI*meanR + c1) * (2.0*covIR + c2)) / ((meanI**2.0 + meanR**2.0 + c1) * (varI + varR + c2))
    return SSIM


def calc_ssim(image, reference, bbox=None, direct=True, imagekey=None):
    assert image.shape == reference.shape
    if direct:
        return ssim_direct(image, reference, bbox)
    else:
        return ssim_fast(image, reference, bbox, imagekey)    
    
if __name__ == "__main__":
    import osgeo.gdal as gdal
    dsImage = gdal.Open(sys.argv[1])
    dsReference = gdal.Open(sys.argv[2])
    try:
        fast = sys.argv[3] == '--fast'
    except:
        fast = False
    image = dsImage.GetRasterBand(1).ReadAsArray()
    reference = dsReference.GetRasterBand(1).ReadAsArray()
    print "SSIM is: ", ssim(image, reference, direct=(not fast))
