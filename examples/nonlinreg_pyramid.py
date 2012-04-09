""" 
Estimates a linear warp field, using an image pyramid to speed up the 
computation.
"""

import numpy as np
import scipy.ndimage as nd
import scipy.misc as misc

from imreg.models import model
from imreg.metrics import metric
from imreg.samplers import sampler
from imreg.visualize import plot
from imreg import register

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
spline = register.Register(
    model.CubicSpline,
    metric.Residual,
    sampler.Spline
    )

# Image pyramid registration can be executed like so:
fullSearch = []
displacement = None

for factor in [ 20., 10.,  5.]:
    
    downImage = image.downsample(factor) 
    downTemplate = template.downsample(factor) 
    
    step, search = spline.register(
        downImage,
        downTemplate,
        displacement=displacement,
        verbose=True
        )
    
    displacement = step.displacement
    
    fullSearch.extend(search)
    
plot.searchInspector(fullSearch)
