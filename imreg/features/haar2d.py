import numpy as np
import scipy.ndimage as nd
import math

__debug=False

def _debug(something):
    global __debug
    if __debug:
        print something

def haar2d(image, levels, debug=False):
    """
    2D Haar wavelet decomposition for levels=levels.
    
    Parameters
    ----------
    image: nd-array
        Input image 
    levels: int
        Number of wavelet levels to compute
    debug:
        Setting debug=True will produce some debug output messages

    Returns
    -------
    haarImage: nd-array
       An image containing the Haar decomposition of the input image. 
       Might be larger than the input image.

    See also
    --------
    register.features.ihaar2d
    """
    global __debug 
    __debug = debug
    assert len(image.shape) == 2, 'Must be 2D image!'
    origRows, origCols = image.shape
    extraRows = 0;
    extraCols = 0;
    while (((origRows + extraRows) >> levels) << levels != (origRows + extraRows)):
        extraRows += 1
    while (((origCols + extraCols) >> levels) << levels != (origCols + extraCols)):
        extraCols += 1
    _debug("Padding: %d x %d -> %d x %d" % (origRows, origCols, origRows + extraRows, origCols + extraCols))

    # Pad image to compatible shape using repitition
    rightFill = np.repeat(image[:, -1:], extraCols, axis=1)
    _image = np.zeros([origRows, origCols + extraCols])
    _image[:, :origCols] = image
    _image[:, origCols:] = rightFill
    bottomFill = np.repeat(_image[-1:, :], extraRows, axis=0)
    image = np.zeros([origRows + extraRows, origCols + extraCols])
    image[:origRows, :] = _image
    image[origRows:, :] = bottomFill
    _debug("Padded image is: %d x %d" % (image.shape[0], image.shape[1]))

    haarImage = image
    for level in range(1,levels+1):
        halfRows = image.shape[0] / 2 ** level
        halfCols = image.shape[1] / 2 ** level
        _image = image[:halfRows*2, :halfCols*2]
        # rows
        lowpass = (_image[:, :-1:2] + _image[:, 1::2]) / 2
        higpass = (_image[:, :-1:2] - _image[:, 1::2]) / 2
        _image[:, :_image.shape[1]/2] = lowpass
        _image[:, _image.shape[1]/2:] = higpass
        # cols
        lowpass = (_image[:-1:2, :] + _image[1::2, :]) / 2
        higpass = (_image[:-1:2, :] - _image[1::2, :]) / 2
        _image[:_image.shape[0]/2, :] = lowpass
        _image[_image.shape[0]/2:, :] = higpass
        haarImage[:halfRows*2, :halfCols*2] = _image    

    _debug(haarImage)
    return haarImage

def ihaar2d(image, levels, debug=False):
    """
    2D Haar wavelet decomposition inverse for levels=levels.
    
    Parameters
    ----------
    image: nd-array
        Input image 
    levels: int
        Number of wavelet levels to de-compute
    debug:
        Setting debug=True will produce some debug output messages

    Returns
    -------
    image: nd-array
       An image containing the inverse Haar decomposition of the input image. 

    See also
    --------
    register.features.haar2d
    """    
    global __debug 
    __debug = debug
    assert len(image.shape) == 2, 'Must be 2D image!'
    origRows, origCols = image.shape
    extraRows = 0;
    extraCols = 0;
    while (((origRows + extraRows) >> levels) << levels != (origRows + extraRows)):
        extraRows += 1
    while (((origCols + extraCols) >> levels) << levels != (origCols + extraCols)):
        extraCols += 1
    assert (extraRows, extraCols) == (0,0), 'Must be compatible shape!'

    for level in range(levels, 0, -1):
        _debug("level=%d" % level)
        halfRows = image.shape[0] / 2 ** level
        halfCols = image.shape[1] / 2 ** level
        # cols
        lowpass = image[:halfRows*2, :halfCols].copy()
        higpass = image[:halfRows*2, halfCols:halfCols*2].copy()
        image[:halfRows*2, :halfCols*2-1:2] = lowpass + higpass 
        image[:halfRows*2, 1:halfCols*2:2] = lowpass - higpass
        _debug(image)
        # rows
        lowpass = image[:halfRows, :halfCols*2].copy()
        higpass = image[halfRows:halfRows*2, :halfCols*2].copy()
        image[:halfRows*2-1:2, :halfCols*2] = lowpass + higpass
        image[1:halfRows*2:2, :halfCols*2] = lowpass - higpass

    return image
