""" 
Estimates a linear warp field, using an image pyramid to speed up the 
computation.
"""

import numpy as np
import scipy.ndimage as nd
import scipy.misc as misc

from register.models import model
from register.metrics import metric
from register.samplers import sampler
from register.visualize import plot
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
affine = register.Register(
    model.CubicSpline,
    metric.Residual,
    sampler.Spline
    )

fullSearch = []

# Image pyramid registration can be executed like so:
warp = None

for factor in [ 10.,  5.]:
    
    if warp is not None:
        scale = downImage.coords.spacing / factor
        # FIXME: Find a nicer way to do this.
        
        warp = np.array(
            [scale*nd.zoom(warp[0], scale), scale*nd.zoom(warp[1], scale)]
            )
        
    downImage = image.downsample(factor) 
    downTemplate = template.downsample(factor) 
    
    step, search = affine.register(
        downImage,
        downTemplate,
        warp=warp,
        verbose=True
        )
    
    warp = step.warp
    
    fullSearch.extend(search)
    
plot.searchInspector(fullSearch)
