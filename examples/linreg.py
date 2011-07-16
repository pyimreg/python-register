import scipy.ndimage as nd
import scipy.misc as misc

from register import register
from register.visualize import plot

image = misc.lena()
image = nd.zoom(image, 0.20)

template = nd.rotate(image, 20, reshape=False)

image = register.smooth(image, 1.5)
template = register.smooth(template, 1.5)

affine = register.register(
    model='affine',
    sampler='nearest'
    )

p, warp, img, error = affine.register(
    image, 
    template,
    plotCB=plot.gridPlot
    )

plot.show()