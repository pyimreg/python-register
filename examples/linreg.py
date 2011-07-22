import scipy.ndimage as nd
import scipy.misc as misc

from register.models import model
from register.metrics import metric
from register.samplers import sampler

from register.visualize import plot
from register import register

image = misc.lena()
image = nd.zoom(image, 0.20)

template = nd.rotate(image, 20, reshape=False)

image = register.smooth(image, 1.5)
template = register.smooth(template, 1.5)

affine = register.Register(
    model.Affine,
    metric.Residual,
    sampler.Nearest
     )

p, warp, img, error = affine.register(
    image,
    template,
    plotCB=plot.gridPlot
    )

plot.show()
