import numpy as np
import register.samplers.sampler as sampler
import time

from register.features.haar2d import haar2d, ihaar2d

def test_haar2d():
    """
    Asserts that some basic test cases are correct.
    
    """

    assert haar2d(np.random.random([5,3]),2,debug=True).shape == (8,4), "Transform data must be padded to compatible shape."
    assert haar2d(np.random.random([8,4]),2,debug=True).shape == (8,4), "Transform data must be padded to compatible shape, only if neccersary."

    image = np.random.random([5,3])
    haart = haar2d(image, 3, debug=False)
    haari = ihaar2d(haart, 3, debug=False)[:image.shape[0], :image.shape[1]]
    assert (image - haari < 0.0001).all(), "Transform must be circular."
