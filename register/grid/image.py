""" A container class for image data """

import numpy as np
import collections

GridSpec = collections.namedtuple(
    'grid',
    'units domain homogeneous tensor'
    )

class Image(object):
    """
    A container for registration data.
    """
    def __init__(self, data, grid=None):

        self.data = data
        self.grid = grid if grid else self.__defaultGrid()

    def __defaultGrid(self):
        """
        Estimates the default coordinates for an "Image" based on the
        dimensions of the image data.
        """

        units = 'pixels (au)'
        domain = [0, self.data.shape[0], 0, self.data.shape[1]]
        tensor = np.mgrid[0.:domain[1], 0.:domain[3]]

        homogenous = np.zeros((3,self.grid[0].size))

        homogenous[0] = self.grid[1].flatten()
        homogenous[1] = self.grid[0].flatten()
        homogenous[2] = 1.0

        return GridSpec(units, domain, homogenous, tensor)

    @staticmethod
    def resample(self, image):
        """
        Returns a new Image object resampled to match the coordinates specified.
        """
        raise NotImplementedError
