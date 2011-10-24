""" 
Estimates a nonlinear warp field, using a modified "demons" algorithm.
"""

import numpy as np
import scipy.ndimage as nd
import scipy.misc as misc

from register.models import model
from register.samplers import sampler

from register import register

def warp(image):
    """
    Randomly warps an input image using a cubic spline deformation.
    """
    coords = register.Coordinates(
        [0, image.shape[0], 0, image.shape[1]]
        )
        
    spline_model = model.CubicSpline(coords)
    spline_sampler = sampler.Spline(coords)

    p = spline_model.identity
    #TODO: Understand the effect of parameter magnitude:
    p += np.random.rand(p.shape[0]) * 100 - 50
    
    return spline_sampler.f(image, spline_model.warp(p)).reshape(image.shape)


# Form some test data (lena, lena rotated 20 degrees)
image = register.RegisterData(misc.lena())
template = register.RegisterData(warp(misc.lena()))

# Form the registrator.
demon = register.DemonsRegister(
    sampler.Nearest
    )

# Image pyramid registration can be executed like so:
downImage = image.downsample(2) 
downTemplate = template.downsample(2) 
    
result = demon.register(
    downImage,
    downTemplate,
    )