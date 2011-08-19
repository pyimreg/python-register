
import collections

import numpy as np
import scipy as sp

from register import register

from register.models import model
from register.metrics import metric
from register.samplers import sampler

import time
 
def print_timing(func):
    def wrapper(*arg, **kwargs):
        t1 = time.time()
        res = func(*arg, **kwargs)
        t2 = time.time()
        print '%s took %0.3f ms' % (func.func_name, (t2-t1)*1000.0)
        return res
    return wrapper


@print_timing
def warp(A, B, multiStage=False, samples=10):
    
    # Form the shift registration instance.
    shift = register.Register(
        model.Shift,
        metric.Residual,
        sampler.Spline
        )
    
    # Form the shift registration instance.
    affine = register.Register(
        model.Affine,
        metric.Residual,
        sampler.Spline
        )
    
    # Coerce the image data into RegisterData.
    image = register.RegisterData(A)
    template = register.RegisterData(B)
    
    # Register (forwards)
    p, warp, img, error = shift.register(
        image,
        template,
        )
    
    if multiStage:
        p, warp, img, error = affine.register(
            image, 
            template,
            p = [0., 0., 0.0, 0., p[0], p[1]],
            )
    
    p, inv_warp, img, error = shift.register(
        template,
        image,
        )
    
    if multiStage:
        p, inv_warp, img, error = affine.register(
            template,
            image,
            p = [0., 0., 0.0, 0., p[0.], p[1]],
            )
        
    def scaledWarp(warp, tensor, scale):
        swarp = warp.copy()
        swarp[0] = tensor[0] + scale*(warp[0] - tensor[0])
        swarp[1] = tensor[1] + scale*(warp[1] - tensor[1])
        return swarp
    
    
    splineSampler = sampler.Spline(image.coords)
    
    advection =  collections.namedtuple(
        'advected',
        'forward inverse warp invWarp scale'
        )
    
    advections = []
    
    for scale in np.linspace(0.0, 1.0, samples):
        
        scaledForward = scaledWarp(warp, image.coords.tensor, scale)
        scaledInverse = scaledWarp(inv_warp, image.coords.tensor, (1.0-scale))
        
        advections.append(
            advection(
                splineSampler.f(image.data, scaledForward).reshape(image.data.shape),
                splineSampler.f(template.data, scaledInverse).reshape(image.data.shape),
                scaledForward,
                scaledInverse,
                scale
                )
            )
    
    return advections