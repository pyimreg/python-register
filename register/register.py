""" A top level registration module """

import collections

import grid.coordinates as coordinates

from samplers import sampler
from models import model
from metrics import metric

import numpy as np
import scipy.ndimage as nd

SAMPLERS = {
    'spline': sampler.Spline,
    'nearest': sampler.Nearest
           }

METRICS = {
    'residual': metric.Residual,
          }

MODELS = {
    'shift': model.Shift,
    'affine': model.Affine,
    'spline': model.Spline
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
    A registration class for estimating the deformation model parameters that
    best solve:

    f( W(I;p), T )

    where:
        f     : is a similarity metric.
        W(x;p): is a deformation model (defined by the parameter set p).
        I     : is an input image (to be deformed).
        T     : is a template (which is a deformed version of the input).

    """

    optStep = collections.namedtuple('optStep', 'error p deltaP')

    def __init__(self,
                 model='shift',
                 metric='residual',
                 sampler='nearest',
                ):

        self.model = MODELS[model]
        self.metric = METRICS[metric]
        self.sampler = SAMPLERS[sampler]

    def __deltaP(self, J, e, alpha, p=None):
        """
        Compute the parameter update.

        Refer to the Levernberg-Marquardt algorithm:
            http://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm

        @param J: dE/dP the relationship between image differences and model
                  parameters.
        @param e: the difference between the image and template.
        @param alpha: the dampening factor.
        @keyword p: the current parameter set.
        @return: deltaP, the set of model parameter updates. (p x 1).
        """

        H = np.dot(J.T, J)

        H += np.diag(alpha*np.diagonal(H))

        return np.dot( np.linalg.inv(H), np.dot(J.T, e))

    def __dampening(self, alpha, decreasing):
        """
        Returns the dampening value.

        Refer to the Levernberg-Marquardt algorithm:
            http://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm

        @param alpha: a dampening factor.
        @param decreasing: a boolean indicating that the error function is
        decreasing.
        @return: an adjusted dampening factor.
        """
        return alpha / 10. if decreasing else alpha * 10.

    def register(self,
                 image,
                 template,
                 p=None,
                 warp=None,
                 regularize=False,
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

        coords = coordinates.Coordinates()
        coords.form(image.shape)

        model = self.model(coords)
        metric = self.metric()
        sampler = self.sampler(coords)


        if warp is not None:
            # Estimate p, using the warp field.
            p = model.estimate(warp)

        p = model.identity if p is None else p
        deltaP = np.zeros_like(p)

        search = []
        alpha = 1e-4
        decreasing = True
        badSteps = 0

        for itteration in range(0,MAX_ITER):

            # Compute the warp field (warp field is the inverse warp)
            warp = model.warp(p)

            # Sample the image using the inverse warp.
            warpedImage = smooth(sampler.f(image, warp).reshape(image.shape), 0.5)

            # Evaluate the error metric.
            e = metric.error(warpedImage, template)

            searchStep = self.optStep(error=np.abs(e).sum(),
                                      p=p,
                                      deltaP=deltaP,
                                      )

            if (len(search) > 1):

                decreasing = (searchStep.error < search[-1].error)

                alpha = self.__dampening(
                    alpha,
                    decreasing
                    )

                if decreasing:

                    if plotCB is not None:
                        plotCB(image,
                               template,
                               warpedImage,
                               coords.grid,
                               warp,
                               '{}:{}'.format(model.MODEL, itteration)
                               )
                else:
                    badSteps += 1

                    if badSteps > MAX_BAD:
                        if verbose:
                            print ('Optimization break, maximum number '
                                   'of bad iterations exceeded.')
                        break

                    # Restore the parameters from the previous iteration.
                    p = search[-1].p
                    continue

            # Computes the derivative of the error with respect to model
            # parameters.

            J = metric.jacobian(model, warpedImage)

            deltaP = self.__deltaP(
                J,
                e,
                alpha,
                p=p
                )

            # Evaluate stopping condition:
            if np.dot(deltaP.T, deltaP) < 1e-4:
                break

            p += deltaP

            if verbose and decreasing:
                print ('{0}\n'
                       'iteration  : {1} \n'
                       'parameters : {2} \n'
                       'error      : {3} \n'
                       '{0}\n'
                      ).format(
                            '='*80,
                            itteration,
                            ' '.join( '{:3.2f}'.format(param) for param in searchStep.p),
                            searchStep.error
                            )

            # Append the search step to the search.
            search.append(searchStep)

        return p, warp, warpedImage, searchStep.error



class KybicRegister(Register):

    def __init__(self,
                 model='shift',
                 metric='residual',
                 sampler='nearest',
                ):

        Register.__init__(self.model, self.metric, self.sampler)

    def __deltaP(self, J, e, alpha):
        """
        Compute the parameter update.
        """

        H = np.dot(J.T, J)

        H += np.diag(alpha*np.diagonal(H))

        return np.dot( np.linalg.inv(H), np.dot(J.T, e))

    def __dampening(self, alpha, decreasing):
        """
        Returns the dampening value.

        Refer to the Levernberg-Marquardt algorithm:
            http://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm

        @param alpha: a dampening factor.
        @param decreasing: a boolean indicating that the error function is
        decreasing.
        @return: an adjusted dampening factor.
        """
        return alpha
