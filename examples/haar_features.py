""" 
Detects haar salient features in an image - 
    
"""

import numpy as np
import scipy.ndimage as nd
import scipy.misc as misc
from matplotlib.pyplot import imread
 
from register import register
from register.features import features
from register.visualize import plot

# Load the image.
image = imread('data/cameraman.tif').astype(np.double)
#image = nd.zoom(image, 0.50)

detector = features.HaarDetector(levels=4, maxpoints=200)
features = detector.detect(image)
image = register.RegisterData(image, features=features)

plot.featurePlotSingle(image)
plot.show()
