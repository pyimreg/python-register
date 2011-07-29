""" 
Estimates a linear warp field, between two images - 

The target is a "smile" the image is a frown and the goal is to estimate
the warp field between them.
    
The deformation is not manufactured by the spline model, and is a good
(realistic) test of the spline deformation model.
    
"""

from matplotlib.pyplot import imread

from register.models import model
from register.metrics import metric
from register.samplers import sampler

from register.visualize import plot
from register import register

# Form some test data (lena, lena rotated 20 degrees)
image = imread('data/frown.png')[:, :, 0]
template = imread('data/smile.png')[:, :, 0]

# Form the affine registration instance.
affine = register.Register(
    model.Spline,
    metric.Residual,
    sampler.CubicConvolution
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
