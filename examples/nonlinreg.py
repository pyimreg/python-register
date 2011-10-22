""" 
Estimates a non-linear warp field, the lenna image is randomly deformed using
the spline deformation model.
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


image = misc.lena()
template = warp(image)

# Coerce the image data into RegisterData.
image = register.RegisterData(image).downsample(5.0)
template = register.RegisterData(template).downsample(5.0)

# Form the affine registration instance.
affine = register.Register(
    model.Affine,
    metric.Residual,
    sampler.Spline
    )


# Form the spline registration instance.
spline = register.Register(
    model.CubicSpline,
    metric.Residual,
    sampler.Spline
    )

# Compute an affine registration between the template and image.
step, search = affine.register(
    image,
    template,
    )

# Compute a nonlinear (spline) registration, initialized with the warp field
# found using the affine registration.
step, _search = spline.register(
    image,
    template,
    displacement=step.displacement,
    verbose=True
    )

search.extend(_search)

plot.searchInspector(search)
