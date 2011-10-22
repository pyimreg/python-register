import numpy as np

import register.models.model as model
import register.samplers.sampler as sampler
import register.register as register
import scipy.misc as misc
import scipy.ndimage as nd

from matplotlib import pyplot

def test_shift():
    """
    Asserts that the feature point alignment error is sufficiently small.
    """
    
    # Form a dummy coordinate class.
    coords = register.Coordinates(
        [0, 10, 0, 10]
        )
    
    # Form corresponding feature sets.
    p0 = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4,2)
    p1 = p0 + 2.0
    
    shift = model.Shift(coords)
    
    _parameters, error = shift.fit(p0, p1)
    
    print _parameters
    
    # Assert that the alignment error is small.
    assert error <= 1.0, "Unexpected large alignment error : {} grid units".format(error)


def test_affine():
    """
    Asserts that the feature point alignment error is sufficiently small.
    """
    
    # Form a dummy coordinate class.
    coords = register.Coordinates(
        [0, 10, 0, 10]
        )
    
    # Form corresponding feature sets.
    p0 = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4,2)
    p1 = p0 + 2.0
    
    affine = model.Affine(coords)
    
    _parameters, error = affine.fit(p0, p1)
    
    # Assert that the alignment error is small.
    assert error <= 1.0, "Unexpected large alignment error : {} grid units".format(error)


def test_CubicSpline_estimate():
    """
    Asserts that scaling a warp field is a reasonable thing to do.
    """
    
    scale = 2.0
    
    # Form a high resolution image.
    high = register.RegisterData(misc.lena().astype(np.double))
    
    # Form a low resolution image.
    low = high.downsample(scale)
    
    # Make a deformed low resolution image.
    p = model.CubicSpline(low.coords).identity
    p += np.random.rand(p.shape[0]) * 100 - 50
    
    warp = model.CubicSpline(low.coords).transform(p)
    
    dlow = sampler.Nearest(low.coords).f(
        low.data, 
        low.coords.tensor - warp
        ).reshape(low.data.shape)
    
    # Scale the low resolution warp field to the same size as the high resolution 
    # image. 
    
    hwarp = np.array( [nd.zoom(warp[0],scale), nd.zoom(warp[1],scale)] ) * scale
    
    # Estimate the high resolution spline parameters that best fit the 
    # enlarged warp field.
    
    invB = np.linalg.pinv(model.CubicSpline(high.coords).basis)
    
    pHat = np.hstack(
        (np.dot(invB, hwarp[1].flatten()), 
         np.dot(invB, hwarp[0].flatten()))
        )
    
    warpHat = model.CubicSpline(high.coords).warp(pHat)
    
    # Make a deformed high resolution image.
    dhigh = sampler.Nearest(high.coords).f(high.data, warpHat).reshape(high.data.shape)
    
    # down-sample the deformed high-resolution image and assert that the 
    # pixel values are "close".
    dhigh_low = nd.zoom(dhigh, 1.0/scale)
    
    # Assert that the down-sampler highresolution image is "roughly" similar to
    # the low resolution image.
    
    assert (np.abs((dhigh_low[:] - dlow[:])).sum() / dlow.size < 10.0), \
        "Normalized absolute error is greater than 10 pixels."
    
#    import matplotlib.pyplot as plt
#    plt.subplot(1,3,1)
#    plt.imshow(dlow)
#    plt.title('low resolution deformation')
#    plt.subplot(1,3,2)
#    plt.imshow(dhigh_low)
#    plt.title('high resolution deformation')
#    plt.subplot(1,3,3)
#    plt.imshow(dhigh_low-dlow)
#    plt.title('high resolution deformation (down-sampled)')
#    plt.show()


def test_thinPlateSpline():
    """
    Asserts that the feature point alignment error is sufficiently small.
    """
    
    # Form a dummy coordinate class.
    coords = register.Coordinates(
        [0, 10, 0, 10]
        )
    
    # Form corresponding feature sets.
    p0 = np.array([0,1, -1,0, 0,-1, 1, 0]).reshape(4,2)
    p1 = np.array([0,0.75, -1, 0.25, 0, -1.25, 1, 0.25]).reshape(4,2)
    
    spline = model.ThinPlateSpline(coords)
    
    _parameters, error = spline.fit(p0, p1)
    
    # Assert that the alignment error is small.
    assert error < 1.0, "Unexpected large alignment error."
    
    
def test_thinPlateSplineApproximate():
    """ 
    Asserts that the computed K, P, L and V matrices are formed correctly.
    
    Refer to: 
    
    Bookstein, F. L. (1989). Principal warps: thin-plate splines and the 
    decomposition of deformations. IEEE Transactions on Pattern Analysis 
    and Machine Intelligence, 11(6), 567-585. 
    """
    
    # Form a dummy coordinate class.
    coords = register.Coordinates(
        [0, 100, 0, 100]
        )
    
    # Form corresponding feature sets.
    p0 = np.array([0,1, -1,0, 0,-1, 1, 0]).reshape(4,2)
    p1 = np.array([0,0.75, -1, 0.25, 0, -1.25, 1, 0.25]).reshape(4,2)
    
    spline = model.ThinPlateSpline(coords)
    
    _parameters, _error, L = spline.fit(p0, p1, lmatrix=True)
    
    # This expected L matrix is derived from the symmetric example in the 
    # referenced paper.
    expectedL = np.array(
        [[ 0.    , -1.3863, -5.5452, -1.3863,  1.    ,  0.    ,  1.    ],
         [-1.3863,  0.    , -1.3863, -5.5452,  1.    , -1.    ,  0.    ],
         [-5.5452, -1.3863,  0.    , -1.3863,  1.    ,  0.    , -1.    ],
         [-1.3863, -5.5452, -1.3863,  0.    ,  1.    ,  1.    ,  0.    ],
         [ 1.    ,  1.    ,  1.    ,  1.    ,  0.    ,  0.    ,  0.    ],
         [ 0.    , -1.    ,  0.    ,  1.    ,  0.    ,  0.    ,  0.    ],
         [ 1.    ,  0.    , -1.    ,  0.    ,  0.    ,  0.    ,  0.    ]]
        )
    
    assert np.allclose(L, expectedL), \
       "The expected L matrix was not derived."
    
    
