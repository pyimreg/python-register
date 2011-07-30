""" A collection of feature detectors"""

import numpy as np
import scipy.ndimage as nd

from numpy.ctypeslib import load_library
from numpyctypes import c_ndarray

libfeatures = load_library('libfeatures.so', __file__)

class FeatureDetector(object):
    """
    Abstract feature detector.

    @param METHOD: the method implemented by the detector.
    @param DESCRIPTION: a meaningful description of the technique used, with
                        references where appropriate.
    """

    METHOD=None
    DESCRIPTION=None

    def detect(self, image):
        """
        The detection function - provided by the specialized detectors.
        """
        return None

    def __str__(self):
        return 'FeatureDetector: {0} \n {1}'.format(
            self.METHOD,
            self.DESCRIPTION
            )


class HaarDetector(object):
    """
    Haar wavelet feature detector.

    @param METHOD: the method implemented by the detector.
    @param DESCRIPTION: a meaningful description of the technique used, with
                        references where appropriate.
    """

    METHOD="Haar"
    DESCRIPTION="Haar wavelet salient feature detector"

    def __init__(self, levels=4, maxpoints=500):
        self.levels = levels
        self.maxpoints = maxpoints

    def detect(self, image):
        """
        A sampling function, responsible for returning a sampled set of values
        from the given array.

        @param array: an n-dimensional array (representing an image or volume).
        @param coords: array coordinates in cartesian form (n by p).
        """

        if self.levels is None:
            raise ValueError('Levels was not specified.')
        if self.maxpoints is None:
            raise ValueError('MaxPoints was not specified.')

        results = np.zeros([self.maxpoints, 3])
        
        arg0 = c_ndarray(image, dtype=np.float, ndim=2)
        arg1 = c_ndarray(results, dtype=np.int, ndim=2)

        libfeatures.haar(arg0, arg1, int(self.levels))

        features = {'points' : {}}

        for point in results:
            features['points'].__setitem__(point[0], point[1:2])

        return features
