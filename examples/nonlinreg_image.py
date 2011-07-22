from matplotlib.pyplot import imread

from register.models import model
from register.metrics import metric
from register.samplers import sampler

from register.visualize import plot
from register import register

# Load the smile and frown.
image = imread('data/frown.png')[:, :, 0]
template = imread('data/smile.png')[:, :, 0]

# Apply a smoothing kernel to the image and template.
image = register.smooth(image, 2.5)
template = register.smooth(template, 2.5)

spline = register.Register(
    model.Spline,
    metric.Residual,
    sampler.Nearest
    )

# Turn that frown upside down.
p, warp, img, error = spline.register(
    image,
    template,
    alpha=15,
    verbose=True,
    plotCB=plot.gridPlot
    )

plot.show()
