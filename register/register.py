""" A top level registration module """

import grid.coordinates as coordinates

from samplers import sampler
from models import model
from metrics import metric

import numpy as np
import scipy.ndimage as nd

SAMPLERS = { 
    'spline': sampler.spline,
    'nearest': sampler.nearest
           }

METRICS = {
    'residual': metric.residual,
          }

MODELS = {
    'shift': model.shift,
    'affine': model.affine,
    'spline': model.spline
         }

MAX_ITER = 200
MAX_BAD = 20

def smooth(image, variance):
    """
    A simple image smoothing method - using a Gaussian kernel.
    @param image: the input image, a numpy ndarray object.
    @param variance: the width of the smoothing kernel.
    @return: the smoothing input image. 
    """
    return np.real(
        np.fft.ifft2(
            nd.fourier_gaussian(
                np.fft.fft2(image), 
                variance
                               )
                    )
                  )


class Register(object):
    """
    A generic registration class : implemented by following the algorithms 
    described in xxx.
    """
    
    def __init__(self, 
                 model='shift',
                 metric='residual',
                 sampler='nearest'
                ):
        
        self.model = MODELS[model]
        self.metric = METRICS[metric]
        self.sampler = SAMPLERS[sampler]
    
    def register(self, 
                 image, 
                 template, 
                 p=None, 
                 warp=None,
                 alpha=10,
                 plotCB=None,
                 verbose=False):
        """
        Performs an image registration.
        
        @param p: a list of parameters, (first guess).
        @param floating: a floating image, numpy ndarray.
        @param target: a target image, numpy ndarray.
        @keyword mask: an optional parameter mask.
        """
        
        coords = coordinates.coordinates()
        coords.form(image.shape)
        
        model = self.model(coords)
        metric = self.metric()
        sampler = self.sampler(coords)
        
        
        if warp is not None:
            # Estimate p, using the warp field.
            p = model.estimate(warp)
            
            
        p = model.identity if p is None else p
        
        
        error = []
        lastP = p
        scale = 1e-4
        badSteps = 0
        
        for step in range(0,MAX_ITER):
            
            # Compute the warp field (warp field is the inverse warp)
            warp = model.warp(p)
            
            # Sample the image using the inverse warp.
            warpedImage = sampler.f(image, warp).reshape(image.shape)
            
            # Evaluate the error metric.
            e = metric.error(warpedImage, template)
            
            error.append(e.sum())
        
            if plotCB is not None:
                plotCB(image, 
                       template, 
                       warpedImage, 
                       coords.grid,
                       warp, 
                       step)
            
            if (step > 1):
                if ( error[-1] < error[-2]):
                    scale /= alpha
                    lastP = p
                    
                    if plotCB is not None:
                        plotCB(image, 
                               template, 
                               warpedImage, 
                               coords.grid,
                               warp, 
                               step)
                    
                else:
                    error = error[0:-1]
                    p = lastP
                    scale *= alpha
                    badSteps += 1
                    
                    if badSteps > MAX_BAD:
                        if verbose:
                            print ('Optimization break, maximum number' 
                                   'of bad iterations exceeded.')
                        break
                    
            # Computes the derivative of the error with respect to model
            # parameters. 
            
            J = metric.jacobian(model, warpedImage)
            
            H = np.dot(J.T, J) 
            
            H += np.diag(scale*np.diagonal(H))
            
            deltaP = np.dot( np.linalg.inv(H), np.dot(J.T, e)) 
                        
            if np.dot(deltaP.T, deltaP) < 1e-4:
                break
            
            p += deltaP
            
            if verbose:
                print ('{0}\n'
                       'iteration  : {1} \n'
                       'parameters : {2} \n'
                       'error      : {3} \n' 
                       '{0}\n'
                      ).format( 
                            '='*80,
                            step,
                            ' '.join( '{:3.2f}'.format(param) for param in p),
                            error[-1]
                            )
                
        return p, warp, warpedImage, error

