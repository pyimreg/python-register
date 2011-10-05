import time

import numpy as np
import scipy as sp

import register.models.model as model
import register.samplers.sampler as sampler

from register import register


def test_sampler():
    """
    Asserts that NN < Cubic < Spline, over a range of image resolutions.
    
    If (one day) something really amazing happens and the scipy map_coordiantes
    method is significantly faster we could favour that as a default.
    
    """
    
    for n in range(128, 2024, 128):
        coords = register.Coordinates(
            [0, n, 0, n]
            )
        
        img  = np.random.rand(n,n)
        warp = np.random.rand(2,n,n)
        
        # nearest neighbour sampler - ctypes
        nearest = sampler.Nearest(coords)
        
        ntimes = np.zeros(10)
        for i in range(0,10):
            t1 = time.time()
            nearest.f(img, warp)
            t2 = time.time()
            ntimes[i] = (t2-t1)*1000.0
        
        print 'Nearest : {0}x{0} image - {1:0.3f} ms'.format(n, np.average(ntimes))
        
        # cubic convolution sampler - ctypes
        cubic = sampler.CubicConvolution(coords)
        
        ctimes = np.zeros(10)
        for i in range(0,10):
            t1 = time.time()
            cubic.f(img, warp)
            t2 = time.time()
            ctimes[i] = (t2-t1)*1000.0
        
        print 'Cubic : {0}x{0} image - {1:0.3f} ms'.format(n, np.average(ctimes))

        # spline sampler - scipy buffered? ctypes?
        spline = sampler.Spline(coords)
        
        stimes = np.zeros(10)
        for i in range(0,10):
            t1 = time.time()
            spline.f(img, warp)
            t2 = time.time()
            stimes[i] = (t2-t1)*1000.0
        
        print 'Spline : {0}x{0} image - {1:0.3f} ms'.format(n, np.average(stimes))
        print '===================================='
        
        assert np.average(ntimes) < np.average(ctimes)
        assert np.average(ntimes) < np.average(stimes)
        assert np.average(ctimes) < np.average(stimes)



def test_rotate_lenna():
    """
    Warps an image.
    """
    
    
    
    image = register.RegisterData(sp.misc.lena())
    
    p = np.array([0., 0.1, 0.1, 0.0, 0., 0.])
    
    affine = model.Affine(image.coords)
    
    warp = affine.warp(p)
    
    bilinear = sampler.Bilinear(image.coords)
    nearest = sampler.Nearest(image.coords)
    
    resampled_bilinear = bilinear.f(image.data, warp)
    resampled_nearest = nearest.f(image.data, warp)        
    
    import matplotlib.pyplot as plt
    plt.subplot(1,4,1)
    plt.imshow(image.data)
    
    plt.subplot(1,4,2)
    plt.imshow(
        resampled_bilinear.reshape(image.data.shape), 
        vmin=image.data.min(), 
        vmax=image.data.max() 
        )

    plt.subplot(1,4,3)
    plt.imshow(
        resampled_nearest.reshape(image.data.shape), 
        vmin=image.data.min(), 
        vmax=image.data.max() 
        )
    
    plt.subplot(1,4,4)
    plt.imshow(
        resampled_bilinear.reshape(image.data.shape) - resampled_nearest.reshape(image.data.shape), 
        )
    
    plt.show()
    
    assert False
    
    
    
    
    
    
    
