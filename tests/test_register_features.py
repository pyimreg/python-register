import numpy as np

from imreg import model, register
from imreg.samplers import sampler


def test_register():
    """
    Top level registration of a simple unit square.
    """

    img = np.zeros((100,100))
    img[25:75, 25:75] = 1

    image = register.RegisterData(
        img,
        features={
            'points':
                {
                 '001': [25, 25],
                 '002': [25, 75],
                 '003': [75, 25],
                 '004': [75, 75],
                }
            }
        )

    template = register.RegisterData(
        img,
        features={
            'points':
                {
                 '001': [35, 35],
                 '002': [35, 85],
                 '003': [85, 35],
                 '004': [70, 70],
                }
            }
        )

    # Form feature registrator.
    feature = register.FeatureRegister(
        model=model.Shift,
        sampler=sampler.Spline,
        )

    # Perform the registration.
    _p, warp, _img, _error = feature.register(
        image,
        template
        )

    assert not np.allclose(warp, np.zeros_like(warp)), \
        "Estimated warp field is zero."
