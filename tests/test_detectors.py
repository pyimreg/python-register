import os
import matplotlib.pyplot as plt
import register.features.detector as detector 

def test_haardetector():
    """
    Asserts that at-least one feature is detected in the "camera-man" image, 
    which should contain many features.
    """
    
    path = os.path.dirname(__file__)
    
    image = plt.imread('{0}/../examples/data/cameraman.png'.format(path))
    
    options = {
        'levels': 5,
        'threshold': 0.2,
        'locality': 5
        }
    
    features = detector.detect(image, detector.HaarDetector, options)
    
    # Asserts that there are features present:
    
    assert len(features['points'].items()) > 0
    