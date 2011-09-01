import numpy as np

import register.models.model as model
import register.register as register
import register.visualize.plot as plot

def test_thinPlateSpline():
    
    # Form a dummy coordinate class.
    coords = register.Coordinates(
        [0, 10, 0, 10]
        )
    
    # Form corresponding feature sets.
    p0 = np.array([0,1, -1,0, 0,-1, 1, 0]).reshape(4,2)
    p1 = np.array([0,0.75, -1, 0.25, 0, -1.25, 1, 0.25]).reshape(4,2)
    
    spline = model.ThinPlateSpline(coords)
    
    parameters = spline.fit(p0, p1)
    
    warp = spline.warp(parameters)
    
    plot.warpPlot(coords.tensor, coords.tensor, warp)
    
    plot.show()
    
    assert False

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
    
    parameters, L = spline.fit(p0, p1, lmatrix=True)
    
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
    
    