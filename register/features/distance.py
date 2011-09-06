""" Distance metric helper functions for region covariance based feature matchers"""

import numpy as np

def frobeniusNorm(A):
    return np.linalg.norm(A, 'fro')
