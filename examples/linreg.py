import scipy.ndimage as nd
import scipy.misc as misc

from register import register
from register.visualize import matplot

image = misc.lena()
image = nd.zoom(image, 0.20)

#template = nd.shift(image, [20,20])
template = nd.rotate(image, 20, reshape=False)

image = register.smooth(image, 1.5)
template = register.smooth(template, 1.5)

affine = register.Register(
    model='affine',
    sampler='nearest'
    )

p, warp, img, error = affine.register(
    image, 
    template,
    plotCB=matplot.gridPlot
    )

matplot.show()