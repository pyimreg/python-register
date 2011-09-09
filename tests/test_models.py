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

def test_affine_warp():
    # Form a dummy coordinate class.
    image = misc.lena().astype(np.double)
    image = nd.zoom(image, 0.1)
    coords = register.Coordinates([0, image.shape[0]-1, 0, image.shape[1]-1])
    # Create an affine model
    test_model = model.Affine(coords)
    # Initialize model
    p = [0,0,0,0,0,0]
    # Create warp field from model
    warp = test_model.warp(p)
    # Create a sampler
    test_sampler = sampler.Nearest(coords)
    # Warp image using warp field
    warpedImage = test_sampler.f(image, warp).reshape(image.shape)
    # Assert identity model did not warp image
    assert (image - warpedImage <= 1).all(), "Identity model must not warp image."



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
    
    
