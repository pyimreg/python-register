""" 
Detects haar salient features in an image - 
    
"""

import numpy as np
import scipy.ndimage as nd
import scipy.misc as misc
from matplotlib.pyplot import imread, plot, imshow, show
 
from register import register
from register.features import features
from register.features.detector import detect, HaarDetector

# Load the image.
image = imread('data/cameraman.png')
#image = nd.zoom(image, 0.50)

options = {}
options['levels'] = 5         # number of wavelet levels
options['threshold'] = 0.2    # threshold between 0.0 and 1.0 to filter out weak features (0.0 includes all features)
options['locality'] = 5       # minimum (approx) distance between two features  

features = detect(image, HaarDetector, options)

imshow(image, cmap='gray')

for point, saliency in features.items():
    plot(point[1], point[0], 'or')

show()
