""" 
Detects haar salient features in an image - 
    
"""

import numpy as np
import scipy.ndimage as nd
import scipy.misc as misc
 
from register import register
from register.features import features
from register.visualize import plot

# Load the image.
image = misc.lena()
#image = nd.zoom(image, 0.30)

detector = features.HaarDetector(levels=3, maxpoints=100)
print image.shape
features = detector.detect(image)
image = register.RegisterData(image, features)

plot.featurePlot(image)
plot.show()
