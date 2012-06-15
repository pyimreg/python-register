"""
Estimates a linear warp field, using an image pyramid to speed up the
computation.
"""

import scipy.ndimage as nd
import scipy.misc as misc

from imreg import model, metric, register
from imreg.samplers import sampler


# Form some test data (lena, lena rotated 20 degrees)
image = register.RegisterData(misc.lena())
template = register.RegisterData(
    nd.rotate(image.data, 20, reshape=False)
    )

# Form the registrator.
affine = register.Register(
    model.Affine,
    metric.Residual,
    sampler.CubicConvolution
    )

fullSearch = []

# Image pyramid registration can be executed like so:
pHat = None
for factor in [30., 20. , 10., 5., 2., 1.]:

    if pHat is not None:
        scale = downImage.coords.spacing / factor
        # FIXME: Find a nicer way to do this.
        pHat = model.Affine.scale(pHat, scale)

    downImage = image.downsample(factor)
    downTemplate = template.downsample(factor)

    step, search = affine.register(
        downImage,
        downTemplate,
        p=pHat,
        verbose=True
        )

    pHat = step.p

    fullSearch.extend(search)
