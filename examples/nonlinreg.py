import numpy as np
import scipy.ndimage as nd
import scipy.misc as misc

from register import register
from register.visualize import plot

def warp(image):
    """
    Randomly warps an input image using a cubic spline deformation.
    """
    
    import register.grid.coordinates as coordinates
    import register.models.model as model
    import register.samplers.sampler as sampler
    
    coords = coordinates.coordinates()
    coords.form(image.shape)
    
    spline_model = model.spline(coords)
    spline_sampler = sampler.spline(coords) 
    
    p = spline_model.identity
    
    #TODO: Understand the effect of parameter magnitude:
    p += np.random.rand(p.shape[0]) * 50
    
    return p, spline_sampler.f(image, spline_model.warp(p)).reshape(image.shape)


image = misc.lena()
image = nd.zoom(image, 0.40)
_p, template = warp(image)

image = register.smooth(image, 1.5)
template = register.smooth(template, 1.5)

# Estimate the affine warp field - use that to initialize the spline.

affine = register.Register(
    model='affine',
    sampler='spline'
    )

spline = register.Register(
    model='spline',
    sampler='spline'
    )

p, warp, img, error = affine.register(
    image, 
    template,
    alpha=10,
    plotCB=plot.gridPlot
    )

p, warp, img, error = spline.register(
    image, 
    template,
    alpha=15,
    warp=warp,
    verbose=True,
    plotCB=plot.gridPlot
    )

plot.show()