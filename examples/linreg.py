import scipy.ndimage as nd
import scipy.misc as misc

from scikits.morph import register
from scikits.morph.visualize import matplot

image = misc.lena()
image = nd.zoom(image, 0.30)

#template = nd.shift(image, [20,20])
template = nd.rotate(image, 20, reshape=False)

image = register.smooth(image, 0.5)
template = register.smooth(template, 0.5)

affine = register.Register(
    model='affine',
    sampler='spline'
    )

p, img, error = affine.register(
    image, 
    template,
    plotCB=matplot.gridPlot
    )

matplot.show()