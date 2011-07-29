import numpy as np
import register.samplers.sampler as sampler
import time

from register import register

def test_sampler():
    """
    Simple test to demonstrate the slowness of samplers.
    """
    
    for n in range(128, 2024, 64):
        coords = register.Coordinates(
            [0, n, 0, n]
            )
        
        img  = np.random.rand(n,n)
        warp = np.random.rand(2,n,n)
        
        # nearest neighbour sampler - ctypes
        nearest = sampler.Nearest(coords)
        
        times = np.zeros(10)
        for i in range(0,10):
            t1 = time.time()
            nearest.f(img, warp)
            t2 = time.time()
            times[i] = (t2-t1)*1000.0
        
        print 'Nearest : {0}x{0} image - {1:0.3f} ms'.format(n, np.average(times))
        
        # spline sampler - scipy buffered? ctypes?
        spline = sampler.Spline(coords)
        
        times = np.zeros(10)
        for i in range(0,10):
            t1 = time.time()
            spline.f(img, warp)
            t2 = time.time()
            times[i] = (t2-t1)*1000.0
        
        print 'Spline : {0}x{0} image - {1:0.3f} ms'.format(n, np.average(times))
        
    assert False
