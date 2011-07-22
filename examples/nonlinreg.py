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
    import register.grid.coordinates as coordinates
    
    coords = coordinates.Coordinates()
    coords.form(image.shape)

    spline_model = model.Spline(coords)
    spline_sampler = sampler.Spline(coords)

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
    model.Affine,
    metric.Residual,
    sampler.Nearest
    )

spline = register.Register(
    model.Spline,
    metric.Residual,
    sampler.Spline
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
