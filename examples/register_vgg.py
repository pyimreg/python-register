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
from os.path import basename

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
image.smooth(0.0001)
template.smooth(0.0001)

# Register.
print "Registering..."
p, warp, img, error = affine.register(
    image,
    template,
#    p=[0,0,0,0,0,0],
#    plotCB=plot.gridPlot,
    verbose=True
    )

#print "Close dialog to exit..."
#plot.show()

pyplot.imsave('png/%s.png' % basename(sys.argv[1])[5:8], image.data, cmap='gray', format='png') 
pyplot.imsave('png/%s.png' % basename(sys.argv[2])[5:8], template.data, cmap='gray', format='png') 

Hfile = open('H/%s.%s.H' % (basename(sys.argv[1])[5:8], basename(sys.argv[2])[5:8]), 'w')
Hfile.write('%f,%f,%f\n%f,%f,%f\n0.0,0.0,1.0\n' % (p[0], p[2], p[4], p[1], p[3], p[5]))
Hfile.close()

print "Done."
