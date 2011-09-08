""" 
Estimates a linear warp field, the target is a file passed as parameter.

Output is placed in VGG directory structure as used by Supreme library:
http://mentat.za.net/supreme

"""

import scipy.ndimage as nd
import scipy.misc as misc

from register.models import model
from register.metrics import metric
from register.samplers import sampler

from register.visualize import plot
from register import register

from matplotlib import pyplot
import osgeo.gdal as gdal
import sys

print "Loading images..."
dsImage = gdal.Open(sys.argv[1])
dsTemplate = gdal.Open(sys.argv[2])
image = dsImage.GetRasterBand(1).ReadAsArray()
template = dsTemplate.GetRasterBand(1).ReadAsArray()

# Form the affine registration instance.
print "Setting up affine registration..."
affine = register.Register(
    model.Affine,
    metric.Residual,
    sampler.CubicConvolution
    )

# Coerce the image data into RegisterData.
print "Loading images into RegisterData objects..."
image = register.RegisterData(image)
template = register.RegisterData(template)

# Smooth the template and image.
image.smooth(0.002)
template.smooth(0.002)

# Register.
print "Registering..."
p, warp, img, error = affine.register(
    image,
    template,
    alpha=0.000001,
    plotCB=plot.gridPlot,
    verbose=True
    )

print "Close dialog to exit..."
plot.show()

print "Done."
