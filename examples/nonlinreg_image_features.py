"""
Estimates a warp field, between two images using only image features.

The target is a "smile" the image is a frown and the goal is to estimate
the warp field between them.

The deformation is estimated using thin plate splines and an example of how to
define a custom kernel is shown.
"""

import numpy as np
import yaml

import matplotlib.pyplot as plt

from imreg import model, register
from imreg.samplers import sampler

# Load the image and feature data.
image = register.RegisterData(
    np.average(plt.imread('data/frown.png'), axis=2),
    features=yaml.load(open('data/frown.yaml'))
    )
template = register.RegisterData(
    np.average(plt.imread('data/smile.png'), axis=2),
    features=yaml.load(open('data/smile.yaml'))
    )

###############################################################################
# Using the implementation of thin-plate splines.
###############################################################################

# Form the feature registrator.
feature = register.FeatureRegister(
    model=model.ThinPlateSpline,
    sampler=sampler.Spline,
    )

# Perform the registration.
p, warp, img, error = feature.register(
    image,
    template
    )

print "Thin-plate Spline kernel error: {}".format(error)

###############################################################################
# Defining a custom model and registering features.
###############################################################################

class GaussSpline(model.ThinPlateSpline):
    def __init__(self, coordinates):
        model.ThinPlateSpline.__init__(self, coordinates)

    def U(self, r):
        # Define a gaussian kernel.
        var = 5.0
        return np.exp( -np.power(r,2)/(2*var**2)  )


# Form feature registrator.
feature = register.FeatureRegister(
    model=GaussSpline,
    sampler=sampler.Spline,
    )

# Perform the registration.
p, warp, img, error = feature.register(
    image,
    template
    )

print "Gaussian kernel error: {}".format(error)
