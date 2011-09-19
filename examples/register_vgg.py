""" 
Estimates a linear warp field, the target is a file passed as parameter.

Output is placed in VGG directory structure as used by Supreme library:
http://mentat.za.net/supreme

"""

import scipy.ndimage as nd
import scipy.misc as misc
import numpy as np

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
oimage = dsImage.GetRasterBand(1).ReadAsArray().astype(np.double)
otemplate = dsTemplate.GetRasterBand(1).ReadAsArray().astype(np.double)

# Form the affine registration instance.
print "Setting up affine registration..."
affine = register.Register(
    model.Affine,
    metric.Residual,
    sampler.CubicConvolution
    )

# Coerce the image data into RegisterData.
print "Loading images into RegisterData objects..."
image = register.RegisterData(nd.zoom(oimage, 1.0))
template = register.RegisterData(nd.zoom(otemplate, 1.0))


# Register.
print "Registering..."
p, warp, img, error = affine.register(
    image,
    template,
    alpha=0.0000001,
    #plotCB=plot.gridPlot,
    verbose=True
    )

#print "Close dialog to exit..."
#plot.show()

p[4] = p[4]/1.0
p[5] = p[5]/1.0

filenum = int(basename(sys.argv[1])[5:8])

pyplot.imsave('png/%03d.png' % (filenum), oimage, cmap='gray', format='png') 
pyplot.imsave('png/%03d.png' % (filenum - 1), otemplate, cmap='gray', format='png') 

Hfile = open('H/%03d.%03d.H' % (filenum-1, filenum), 'w')
H = np.linalg.inv(np.array([[p[0]+1.0, p[2], p[4]], [p[1], p[3]+1.0, p[5]], [0.0, 0.0, 1.0]]))
Hfile.write('%f %f %f\n%f %f %f\n%f %f %f\n' % (H[0,0],H[0,1],H[0,2],H[1,0],H[1,1],H[1,2],H[2,0],H[2,1],H[2,2]))
Hfile.close()

p = affine.model(image.coords).identity
warp = affine.model(image.coords).warp(p)
resampledImage = affine.sampler(template.coords).f(image.data, warp).reshape(image.data.shape)
p = affine.model(template.coords).identity
warp = affine.model(template.coords).warp(p)
resampledTemplate = affine.sampler(template.coords).f(template.data, warp).reshape(template.data.shape)

print filenum
print "Before: ", int(np.abs(affine.metric(affine.sampler(None).border).error(resampledImage, resampledTemplate)).sum())
after = int(np.abs(affine.metric(affine.sampler(None).border).error(img, resampledTemplate)).sum())
print "After: ", after

thres = 300000
if after > thres: 
    pyplot.figure()
    pyplot.axis('image')
    pyplot.subplot(1,2,1)
    pyplot.imshow(np.abs(resampledImage-resampledTemplate).astype(np.uint8), cmap='gray')
    pyplot.subplot(1,2,2)
    pyplot.imshow(np.abs(img-resampledTemplate).astype(np.uint8), cmap='gray')
    pyplot.show()
    pyplot.close()

print "Done."
