import numpy as np

import scipy.ndimage as nd
import scipy.misc as misc

import register.grid.coordinates as coordinates
import register.samplers.sampler as sampler
import register.models.model as model

from register import register

def warp(image, p, model, sampler):
    """
    Warps an image given a deformation model a set of parameters.
    
    @param image: an numpy ndarray.
    @param p: warp parameters.
    @param model: a deformation model.
    
    """
    coords = coordinates.coordinates()
    coords.form(image.shape)
    
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
    
    if metafunc.function is test_shift:
        
        for displacement in np.arange(-10.,10.):
        
            p = np.array([displacement, displacement])
            
            template = warp(
                image, 
                p, 
                model.shift,
                sampler.spline
                )
            
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
        
        # test the displacement component
        for displacement in np.arange(-10.,10.):
            
            p = np.array([0., 0., 0., 0., displacement, displacement])
            
            template = warp(
                image, 
                p, 
                model.affine,
                sampler.spline
                )
            
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
        
    
def test_shift(image, template, p):
    """
    Tests image registration using a shift deformation model.
    """
    
    shift = register.register(
        model='shift',
        sampler='spline'
        )
    
    _p, _warp, _img, _error = shift.register(
        register.smooth(image, 0.5),
        register.smooth(template, 0.5)
        )
    
    assert np.allclose(p, _p, atol=0.5), \
        "Estimated p: {} not equal to p: {}".format(
            _p,
            p
            )


def test_affine(image, template, p):
    """
    Tests image registration using a affine deformation model.
    """
    
    affine = register.register(
        model='affine',
        sampler='spline'
        )
    
    _p, _warp, _img, _error = affine.register(
        register.smooth(image, 0.5),
        register.smooth(template, 0.5)
        )
    
    assert np.allclose(p, _p, atol=0.5), \
        "Estimated p: {} not equal to p: {}".format(
            _p,
            p
            )