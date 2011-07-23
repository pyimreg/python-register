""" 
Estimates a linear warp field, the target is a transformed version of lenna:

    http://en.wikipedia.org/wiki/Lenna
"""

import scipy.ndimage as nd
import scipy.misc as misc

from register.models import model
from register.metrics import metric
from register.samplers import sampler

from register.visualize import plot
from register import register

# Form some test data (lena, lena rotated 20 degrees)
image = misc.lena()
image = nd.zoom(image, 0.20)
template = nd.rotate(image, 20, reshape=False)

# Form the affine registration instance.
affine = register.Register(
    model.Affine,
    metric.Residual,
    sampler.Nearest
    )

# Coerce the image data into RegisterData.
image = register.RegisterData(image)
template = register.RegisterData(template)

# Smooth the template and image.
image.smooth(1.5)
template.smooth(1.5)

# Register.
p, warp, img, error = affine.register(
    image,
    template,
    plotCB=plot.gridPlot,
    verbose=True
    )

plot.show()
