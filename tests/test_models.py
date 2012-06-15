import numpy as np

from imreg import model, register
from imreg.samplers import sampler

import scipy.misc as misc
import scipy.ndimage as nd


def test_shift():
    """
    Asserts that the feature point alignment error is sufficiently small.
    """

    # Form a dummy coordinate class.
    coords = register.Coordinates(
        [0, 10, 0, 10]
        )

    # Form corresponding feature sets.
    p0 = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
    p1 = p0 + 2.0

    shift = model.Shift(coords)

    _parameters, error = shift.fit(p0, p1)

    print _parameters

    # Assert that the alignment error is small.
    assert error <= 1.0, "Unexpected large alignment error : {} grid units".format(error)


def test_affine():
    """
    Asserts that the feature point alignment error is sufficiently small.
    """

    # Form a dummy coordinate class.
    coords = register.Coordinates(
        [0, 10, 0, 10]
        )

    # Form corresponding feature sets.
    p0 = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4,2)
    p1 = p0 + 2.0

    affine = model.Affine(coords)

    _parameters, error = affine.fit(p0, p1)

    # Assert that the alignment error is small.
    assert error <= 1.0, "Unexpected large alignment error : {} grid units".format(error)


def test_thinPlateSpline():
    """
    Asserts that the feature point alignment error is sufficiently small.
    """

    # Form a dummy coordinate class.
    coords = register.Coordinates(
        [0, 10, 0, 10]
        )

    # Form corresponding feature sets.
    p0 = np.array([0,1, -1,0, 0,-1, 1, 0]).reshape(4,2)
    p1 = np.array([0,0.75, -1, 0.25, 0, -1.25, 1, 0.25]).reshape(4,2)

    spline = model.ThinPlateSpline(coords)

    _parameters, error = spline.fit(p0, p1)

    # Assert that the alignment error is small.
    assert error < 1.0, "Unexpected large alignment error."


def test_thinPlateSplineApproximate():
    """
    Asserts that the computed K, P, L and V matrices are formed correctly.

    Refer to:

    Bookstein, F. L. (1989). Principal warps: thin-plate splines and the
    decomposition of deformations. IEEE Transactions on Pattern Analysis
    and Machine Intelligence, 11(6), 567-585.
    """

    # Form a dummy coordinate class.
    coords = register.Coordinates(
        [0, 100, 0, 100]
        )

    # Form corresponding feature sets.
    p0 = np.array([0,1, -1,0, 0,-1, 1, 0]).reshape(4,2)
    p1 = np.array([0,0.75, -1, 0.25, 0, -1.25, 1, 0.25]).reshape(4,2)

    spline = model.ThinPlateSpline(coords)

    _parameters, _error, L = spline.fit(p0, p1, lmatrix=True)

    # This expected L matrix is derived from the symmetric example in the
    # referenced paper.
    expectedL = np.array(
        [[ 0.    , -1.3863, -5.5452, -1.3863,  1.    ,  0.    ,  1.    ],
         [-1.3863,  0.    , -1.3863, -5.5452,  1.    , -1.    ,  0.    ],
         [-5.5452, -1.3863,  0.    , -1.3863,  1.    ,  0.    , -1.    ],
         [-1.3863, -5.5452, -1.3863,  0.    ,  1.    ,  1.    ,  0.    ],
         [ 1.    ,  1.    ,  1.    ,  1.    ,  0.    ,  0.    ,  0.    ],
         [ 0.    , -1.    ,  0.    ,  1.    ,  0.    ,  0.    ,  0.    ],
         [ 1.    ,  0.    , -1.    ,  0.    ,  0.    ,  0.    ,  0.    ]]
        )

    assert np.allclose(L, expectedL), \
       "The expected L matrix was not derived."


