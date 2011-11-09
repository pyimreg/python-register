""" 
Estimates a linear warp field, the target is a transformed version of lenna:

    http://en.wikipedia.org/wiki/Lenna
"""

import scipy.ndimage as nd
import scipy.misc as misc

from register.models import model
from register.metrics import metric
from register.samplers import sampler
from register import register

from register.visualize import plot

# Form some test data (lena, lena rotated 20 degrees)
image = misc.lena()
template = nd.rotate(image, 20, reshape=False)

# Form the affine registration instance.
affine = register.CG(
    model.Affine,
    metric.Residual,
    sampler.CubicConvolution
    )

# Coerce the image data into RegisterData.
image = register.RegisterData(image).downsample(2)
template = register.RegisterData(template).downsample(2)

# Register.
step, search = affine.register(
    image,
    template,
    verbose=True,
    plotCB=plot.gridPlot,
    )

plot.show()