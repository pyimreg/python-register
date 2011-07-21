""" A container class for coordinates used by sampling and models """

import numpy as np

class Coordinates(object):

    def __init__(self):

        self.grid = None
        self.homogenous = None

    def form(self, shape):
        """
        Forms both grid and canonical coordinate matrices.

        @param shape: a tuple of array dimensions.
        """

        self.grid = np.mgrid[0.0:shape[0],
                             0.0:shape[1]
                            ]

        self.homogenous = np.zeros((3,self.grid[0].size))

        self.homogenous[0] = self.grid[1].flatten()
        self.homogenous[1] = self.grid[0].flatten()
        self.homogenous[2] = 1.0
