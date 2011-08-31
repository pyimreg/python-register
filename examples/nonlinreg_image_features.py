""" 
Estimates a linear warp field, between two images - 

The target is a "smile" the image is a frown and the goal is to estimate
the warp field between them.
    
The deformation is estimated using thin plate splines.
    
"""

import numpy as np
import yaml 

import matplotlib.pyplot as plt

from register.models import model
from register.samplers import sampler
from register.visualize import plot

from register import register

# Load the image and feature data. 
image = register.RegisterData(
    np.average(plt.imread('data/frown.png'), axis=2),
    features=yaml.load(open('data/frown.yaml'))
    )
template = register.RegisterData(
    np.average(plt.imread('data/smile.png'), axis=2),
    features=yaml.load(open('data/smile.yaml'))
    )


# Form the tps registration instance.
feature = register.FeatureRegister(
    model=model.ThinPlateSpline,
    sampler=sampler.Spline,
    )
    
# Register using features.
p, warp, img = feature.register(
    image,
    template
    )

plot.featurePlot(image, template, img)
plot.show()