import time

import numpy as np

from imreg.samplers import sampler
from imreg import register


def test_sampler():
    """
    Asserts that NN < Bilinear < Cubic < Spline, over a range of image resolutions.

    If (one day) something really amazing happens and the scipy map_coordiantes
    method is significantly faster we could favour that as a default.

    """

    for n in range(128, 1024, 128):
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
        bilinear = sampler.Bilinear(coords)

        btimes = np.zeros(10)
        for i in range(0,10):
            t1 = time.time()
            bilinear.f(img, warp)
            t2 = time.time()
            btimes[i] = (t2-t1)*1000.0

        print 'Bilinear : {0}x{0} image - {1:0.3f} ms'.format(n, np.average(btimes))

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
        assert np.average(ntimes) < np.average(btimes)
        assert np.average(ntimes) < np.average(stimes)
        assert np.average(btimes) < np.average(ctimes)
        assert np.average(ctimes) < np.average(stimes)
