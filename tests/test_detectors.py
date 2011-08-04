import numpy as np
import register.samplers.sampler as sampler
import time
import matplotlib.pyplot as plt
import os

from register.features.detector import detect, HaarDetector

def test_haardetector():
    """
    Excersize the Haar feature detector.
    Asserts that some basic test cases are correct.
    
    """

    path = os.path.dirname(__file__)
    image = plt.imread('%s/../examples/data/cameraman.png' % path)
    
    options = {}
    options['levels'] = 5
    options['threshold'] = 0.2
    options['locality'] = 5

    features = detect(image, HaarDetector, options, debug=True, plt=plt)
    plt.imshow(image, cmap='gray')
    for point, saliency in features.items():
        plt.plot(point[1], point[0], 'or')
    #plt.show()

    
    
if __name__ == '__main__':
    test_haardetector()
