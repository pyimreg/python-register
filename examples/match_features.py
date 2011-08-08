""" 
Detects haar salient features in an image - 
    
"""

import numpy as np
import matplotlib.pyplot as plt
 
from register import register
from register.models import model
from register.samplers import sampler
from register.features.detector import detect, HaarDetector
from register.features import matcher

import register.features.region as region

def warp(image):
    """
    Randomly warps an input image using a cubic spline deformation.
    """
    coords = register.Coordinates(
        [0, image.shape[0], 0, image.shape[1]]
        )
        
    spline_model = model.Spline(coords)
    spline_sampler = sampler.Spline(coords)

    p = spline_model.identity
    #TODO: Understand the effect of parameter magnitude:
    p += np.random.rand(p.shape[0]) * 100 - 50
    
    return spline_sampler.f(image, spline_model.warp(p)).reshape(image.shape)

# Load the image and warp it
reference = plt.imread('data/cameraman.png')
other = warp(reference)


options = {}
options['levels'] = 5         # number of wavelet levels
options['threshold'] = 0.8    # threshold between 0.0 and 1.0 to filter out weak features (0.0 includes all features)
options['locality'] = 5       # minimum (approx) distance between two features  

features = detect(reference, HaarDetector, options)
refData = register.RegisterData(reference, features=features)
#matchedfeatures = matcher.phaseCorrelationMatch(refData, other, chipsize=16, searchsize=64, threshold=0.1)

#plt.show()

plt.subplot(1,2,1)
plt.imshow(reference, cmap='gray')
features['regcov'] = {}
for id, point in features['points'].items():
    C = region.covariance(reference, (point[0]-10, point[1]-10), (point[0]+10, point[1]+10))
    features['regcov'][id] = C
    print C
    plt.plot(point[1], point[0], 'or')

plt.subplot(1,2,2)
plt.imshow(other, cmap='gray')
for id, point in matchedfeatures['points'].items():
    plt.plot(point[1], point[0], 'or')

plt.show()

