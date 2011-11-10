import numpy as np

import scipy.ndimage as nd
import scipy.misc as misc

import register.models.model as model
import register.metrics.metric as metric
import register.samplers.sampler as sampler

from register import register

def warp(image, p, model, sampler):
    """
    Warps an image.
    """
    coords = register.Coordinates(
        [0, image.shape[0], 0, image.shape[1]]
        )
    
    model = model(coords)
    sampler = sampler(coords)

    return sampler.f(image,
                     model.warp(p)
                    ).reshape(image.shape)


def pytest_generate_tests(metafunc):
    """
    Generates a set of test for the registration methods.
    """

    image = misc.lena()
    image = nd.zoom(image, 0.50)

    
    for displacement in np.arange(-10.,10.):

        p = np.array([displacement, displacement])

        template = warp(
            image,
            p,
            model.Shift,
            sampler.Spline
            )

        if metafunc.function is test_shift:
            metafunc.addcall(
                id='dx={}, dy={}'.format(
                    p[0],
                    p[1]
                    ),
                funcargs=dict(
                    image=image,
                    template=template,
                    p=p
                    )
                )

        if metafunc.function is test_affine:
            p = np.array([0., 0., 0., 0., displacement, displacement])
            metafunc.addcall(
                    id='dx={}, dy={}'.format(
                        p[4],
                        p[5]
                        ),
                    funcargs=dict(
                        image=image,
                        template=template,
                        p=p
                        )
                    )
        
        if metafunc.function is test_wrapped:
            p = np.array([0., 0., 0., 0., displacement, displacement])
            metafunc.addcall(
                    id='dx={}, dy={}'.format(
                        p[4],
                        p[5]
                        ),
                    funcargs=dict(
                        image=image,
                        template=template,
                        p=p
                        )
                    )


def test_wrapped(image, template, p):
    """
    Tests image registration using a shift deformation model.
    """

    shift = register.Wrapped(
        model.Shift,
        metric.Residual,
        sampler.Spline
        )

    # Coerce the image data into RegisterData.
    image = register.RegisterData(image)
    template = register.RegisterData(template)
    
    
    step, _search = shift.register(
        image,
        template
        )

    assert np.allclose(p, step.p, atol=0.5), \
        "Estimated p: {} not equal to p: {}".format(
            step.p,
            p
            )



def test_shift(image, template, p):
    """
    Tests image registration using a shift deformation model.
    """

    shift = register.Register(
        model.Shift,
        metric.Residual,
        sampler.Spline
        )

    # Coerce the image data into RegisterData.
    image = register.RegisterData(image)
    template = register.RegisterData(template)
    
    
    step, _search = shift.register(
        image,
        template
        )

    assert np.allclose(p, step.p, atol=0.5), \
        "Estimated p: {} not equal to p: {}".format(
            step.p,
            p
            )


def test_affine(image, template, p):
    """
    Tests image registration using a affine deformation model.
    """

    affine = register.Register(
        model.Affine,
        metric.Residual,
        sampler.Spline
        )

    # Coerce the image data into RegisterData.
    image = register.RegisterData(image)
    template = register.RegisterData(template)
    
    step, _search = affine.register(
        image,
        template
        )

    assert np.allclose(p, step.p, atol=0.5), \
        "Estimated p: {} not equal to p: {}".format(
            step.p,
            p
            )
