""" 
Estimates a linear warp field, the target is a transformed version of lenna:

    http://en.wikipedia.org/wiki/Lenna
"""

import scipy.ndimage as nd
import scipy.misc as misc

from imreg.models import model
from imreg.metrics import metric
from imreg.samplers import sampler
from imreg import register

from imreg.visualize import plot

# Form some test data (lena, lena rotated 20 degrees)
image = misc.lena()
template = nd.rotate(image, 20, reshape=False)

# Form the affine registration instance.
affine = register.Register(
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
    )

# Call the debug tool "searchInspector"
plot.searchInspector(search)
