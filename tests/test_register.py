import numpy as np

import scipy.ndimage as nd
import scipy.misc as misc

import register.grid.coordinates as coordinates
import register.samplers.sampler as sampler
import register.models.model as model

from register import register

def warp(image, p, model, sampler):
    """
    Warps an image given a model a set of parameters.
    
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
    image = nd.zoom(image, 0.25)
    
    if metafunc.function is test_shift:
        
        for p in np.array( [np.arange(-20.,21.), 
                            np.arange(-20.,21.)] ).reshape(41,2):
            
            template = warp(
                image, 
                p, 
                model.shift,
                sampler.spline
                )
            
            metafunc.addcall(
                id='[dx={}, dy={}]'.format(
                    p[0], 
                    p[1]
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
    
    shift = register.Register(
        model='shift',
        sampler='spline'
        )
    
    _p, _warp, _img, _error = shift.register(
        image, 
        template
        )
    
    assert np.allclose(p, _p, atol=1.0), \
        "Estimated p: {} not equal to p: {}".format(
            _p,
            p
            )
    
#def test_affineRegister():
#        
#    affine = register.lsqRegister(
#        model='affine',
#        sampler='nearest'
#        )
#    
#    p, img, error = affine.register(
#        image, 
#        template,
#        verbose=True,
#        plotCB=matplot.simplePlot
#        )
