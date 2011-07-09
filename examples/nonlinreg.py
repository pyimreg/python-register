import numpy as np
import scipy.ndimage as nd
import scipy.misc as misc

from scikits.morph import register
from scikits.morph.visualize import matplot

def warp(image):
    """
    Randomly warps an input image using a cubic spline deformation.
    """
    
    import scikits.morph.grid.coordinates as coordinates
    import scikits.morph.models.model as model
    import scikits.morph.samplers.sampler as sampler
    
    coords = coordinates.coordinates()
    coords.form(image.shape)
    
    spline_model = model.spline(coords)
    spline_sampler = sampler.spline(coords) 
    
    p = spline_model.identity
    p += np.random.rand(p.shape[0]) * 25
    
    return spline_sampler.f(image, spline_model.warp(p)).reshape(image.shape)


image = misc.lena()
image = nd.zoom(image, 0.40)
template = warp(image)

image = register.smooth(image, 1.5)
template = register.smooth(template, 1.5)

spline = register.Register(
    model='spline',
    sampler='spline'
    )


# Estimate the affine warp field - use that to initialize the spline.

affine = register.Register(
    model='affine',
    sampler='spline'
    )


p, warp, img, error = affine.register(
    image, 
    template,
    verbose=True,
    alpha=15,
    plotCB=matplot.gridPlot
    )

p, warp, img, error = spline.register(
    image, 
    template,
    verbose=True,
    alpha=15,
    warp=warp,
    plotCB=matplot.gridPlot
    )

matplot.show()