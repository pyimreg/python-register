import scipy.misc as misc

from imreg import register

def test_downsample():
    """
    Tests register data down-sampling.
    """
    image = register.RegisterData(misc.lena())
    for factor in [1, 2, 4, 6, 8 ,10]:
        subSampled = image.downsample(factor)
        assert subSampled.data.shape[0] == image.data.shape[0] / factor
        assert subSampled.data.shape[1] == image.data.shape[1] / factor
        assert subSampled.coords.spacing == factor
