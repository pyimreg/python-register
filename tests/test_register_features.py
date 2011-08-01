import numpy as np
import time

import register.samplers.sampler as sampler
from register import register


def test_register():
    """ 
    Top level registration of a simple unit square.
    """
    img = np.zeros((100,100))
    img[25:75, 25:75] = 1
    
    image = register.RegisterData(
        img, 
        features={
            'points':
                {
                 '001': [25, 25],
                 '002': [25, 75],
                 '003': [75, 25],
                 '004': [75, 75],
                }
            }
        )         
    
    template = register.RegisterData(
        img, 
        features={
            'points':
                {
                 '001': [35, 35],
                 '002': [35, 85],
                 '003': [85, 35],
                 '004': [50, 50],
                }
            }
        )    
    
    # Form a thinplate spline "registrator"
    
    spline = register.SplineRegister(
        sampler.CubicConvolution
        )
    
    warp, img = spline.register(image, template)
    
    
def test_vectorized():
    """ 
    Asserts that the execution time (on average) of vectorized code is *faster*. 
    """
    
    # Define some dummy data.
    img = np.zeros((100,100))
    img[25:75, 25:75] = 1
    
    image = register.RegisterData(
        img, 
        features={
            'points':
                {
                 '001': [25, 25],
                 '002': [25, 75],
                 '003': [75, 25],
                 '004': [75, 75],
                }
            }
        )         
    
    template = register.RegisterData(
        img, 
        features={
            'points':
                {
                 '001': [35, 35],
                 '002': [35, 85],
                 '003': [85, 35],
                 '004': [50, 50],
                }
            }
        )    
    
    # Form a thinplate spline "registrator"
    
    spline = register.SplineRegister(
        sampler.CubicConvolution
        )
    
    times = np.zeros(10)
    for i in range(0,10):
        t1 = time.time()
        _warp, img = spline.register(image, template)
        t2 = time.time()
        times[i] = (t2-t1)*1000.0
        
    print 'Vectorized : {0}x{0} image - {1:0.3f} ms'.format(
            100, 
            np.average(times)
            )
    
    vtimes = np.zeros(10)
    for i in range(0,10):
        t1 = time.time()
        _warp, img = spline.register(image, template, vectorized=False)
        t2 = time.time()
        vtimes[i] = (t2-t1)*1000.0
        
    print 'Untouched : {0}x{0} image - {1:0.3f} ms'.format(
            100, 
            np.average(vtimes)
            )
    
    assert np.average(times) < np.average(vtimes), \
        "Vectorized code is slower than non-vectorized code. Not good."
    
    assert False
    
def test_approximate():
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
    
    # Form a thinplate spline "registrator"
    
    spline = register.SplineRegister(
        sampler.CubicConvolution(coords)
        )
    
    L, LinvY = spline.approximate(p0, p1)
    
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
    
    