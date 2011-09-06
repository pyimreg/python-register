""" A top level registration module """

import collections
import numpy as np
import scipy.ndimage as nd

from features import region

def _smooth(image, variance):
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


class Coordinates(object):
    """
    A container for grid coordinates.
    """
    def __init__(self, domain, spacing=None):
    
        self.domain = domain
        self.tensor = np.mgrid[0.:domain[1], 0.:domain[3]]
        
        self.homogenous = np.zeros((3,self.tensor[0].size))
        self.homogenous[0] = self.tensor[1].flatten()
        self.homogenous[1] = self.tensor[0].flatten()
        self.homogenous[2] = 1.0


class RegisterData(object):
    """
    A container for registration data.
    """
    def __init__(self, data, coords=None, features=None):

        self.data = data
        
        if not coords:
            self.coords = Coordinates(
                [0, data.shape[0], 0, data.shape[1]]
                )
        else:
            self.coords = coords
        
        # Features are (as a starting point a dictionary) which define
        # labelled salient image coordinates (point features). 
        
        self.features = features
        
        if not self.features is None:
            if not self.features.has_key('featureShape'):
                self.features['featureShape'] = (16,16)
            if not self.features.has_key('regionCovariances'):
                shape = self.features['featureShape']
                features['regionCovariances'] = {}
                for id, point in self.features['points'].items():
                    features['regionCovariances'][id] = region.covariance(data, (point[0]-shape[0]/2, point[1]-shape[1]/2), (point[0]+shape[0]/2, point[1]+shape[1]/2))
        
    def smooth(self, variance):
        """
        A simple image smoothing method - using a Gaussian kernel.
        @param variance: the width of the smoothing kernel.
        """
        self.data = _smooth(self.data, variance)
    

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
    
    # The optimization step cache.
    optStep = collections.namedtuple('optStep', 'error p deltaP')
    
    # The maximum number of optimization iterations. 
    MAX_ITER = 200
    
    # The maximum numver of bad (incorrect) optimization steps.
    MAX_BAD = 20
    
    def __init__(self, model, metric, sampler):

        self.model = model
        self.metric = metric
        self.sampler = sampler

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
                 alpha=None,
                 warp=None,
                 plotCB=None,
                 verbose=False):
        """
        Performs an image registration.
        @param image: the floating image.
        @param template: the target image.
        @keyword p: a list of parameters, (first guess).
        @keyword alpha: the dampening factor.
        @keyword warp: the warp field (first guess).
        @keyword plotCB: a debug plotting function.
        @keyword verbose: a debug flag for text status updates. 
        """
        
        #TODO: Determine the common coordinate system.
        # if image.coords != template.coords:
        #     raise ValueError('Coordinate systems differ.')
            
        # Initialize the models, metric and sampler.
        model = self.model(image.coords)
        sampler = self.sampler(image.coords)
        metric = self.metric()

        if warp is not None:
            # Estimate p, using the warp field.
            p = model.estimate(warp)

        p = model.identity if p is None else p
        deltaP = np.zeros_like(p)

        search = []
        alpha = alpha if alpha is not None else 1e-4
        decreasing = True
        badSteps = 0

        for itteration in range(0,self.MAX_ITER):

            # Compute the warp field (warp field is the inverse warp)
            warp = model.warp(p)
            
            # Sample the image using the inverse warp.
            warpedImage = _smooth(
                sampler.f(image.data, warp).reshape(image.data.shape),
                0.5
                )
            
            # Evaluate the error metric.
            e = metric.error(warpedImage, template.data)

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
                        plotCB(image.data,
                               template.data,
                               warpedImage,
                               image.coords.tensor,
                               warp, 
                               '{0}:{1}'.format(model.MODEL, itteration)
                               )
                else:
                    badSteps += 1

                    if badSteps > self.MAX_BAD:
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
                            ' '.join( '{0:3.2f}'.format(param) for param in searchStep.p),
                            searchStep.error
                            )

            # Append the search step to the search.
            search.append(searchStep)

        return p, warp, warpedImage, searchStep.error


class KybicRegister(Register):
    """
    Variant of LM algorithm as described by:
      
    Kybic, J. and Unser, M. (2003). Fast parametric elastic image
        registration. IEEE Transactions on Image Processing, 12(11), 1427-1442.
    """
    
    def __init__(self, model, metric, sampler):
        Register.__init__(self, model, metric, sampler)

    def __deltaP(self, J, e, alpha, p):
        """
        Compute the parameter update.
        """

        H = np.dot(J.T, J)

        H += np.diag(alpha*np.diagonal(H))

        return np.dot( np.linalg.inv(H), np.dot(J.T, e)) - alpha*p

    def __dampening(self, alpha, decreasing):
        """
        Returns the dampening value, without adjustment.
    
        @param alpha: a dampening factor.
        @param decreasing: a boolean indicating that the error function is
        decreasing.
        @return: an adjusted dampening factor.
        """
        return alpha


class SplineRegister():
    """
    A registration class for estimating the deformation field which minimizes 
    feature differences using a thin-plate-spline interpolant.
    """
    
    def __init__(self, sampler, kernel=None):
        
        self.sampler = sampler
        self.kernel = kernel if kernel is not None else None
        
    def U(self, r):
        """
        This is a kernel function applied to solve the biharmonic equation.
        @param r: 
        """
        
        if not self.kernel:
            return np.multiply( -np.power(r,2), np.log(np.power(r,2) + 1e-20))
        else:
            return self.kernel(r)
        
        ## Gaussian kernel
        ##var = 5.0
        ##return np.exp( -pow(r,2)/(2*var**2)  )
    
    def approximate(self, p0, p1):
        """
        Approximates the thinplate spline coefficients, following derivations
        shown in:
        
        Bookstein, F. L. (1989). Principal warps: thin-plate splines and the 
        decomposition of deformations. IEEE Transactions on Pattern Analysis 
        and Machine Intelligence, 11(6), 567-585. 
        """
        
        K = np.zeros((p0.shape[0], p0.shape[0]))
        
        for i in range(0, p0.shape[0]):
            for j in range(0, p0.shape[0]):
                r = np.sqrt( (p0[i,0] - p0[j,0])**2 + (p0[i,1] - p0[j,1])**2 ) 
                K[i,j] = self.U(r)
                
        P = np.hstack((np.ones((p0.shape[0], 1)), p0))
        
        L = np.vstack((np.hstack((K,P)), 
                       np.hstack((P.transpose(), np.zeros((3,3))))))
        
        Y = np.vstack( (p1, np.zeros((3, 2))) )
        
        Y = np.matrix(Y)
        
        Linv = np.matrix(np.linalg.inv(L))
        
        return L, Linv*Y
    
    def register(self,
                 image,
                 template,
                 vectorized=True):
        """
        Performs an image (feature based) registration.
        
        @param image: a floating image, registerData object.
        @param template: a target image, registerData object.
        """
        
        sampler = self.sampler(image.coords)
        
        # Form corresponding point sets. 
        imagePoints = []
        templatePoints = []
        
        for id, point in image.features['points'].items():
            if id in template.features['points']:
                imagePoints.append(point)
                templatePoints.append(template.features['points'][id])
                #print '{} -> {}'.format(imagePoints[-1], templatePoints[-1])
        
        if not imagePoints or not templatePoints:
            raise ValueError('Requires image and template features to register.')
        
        # Note the inverse warp is estimated here.
        
        p0 = np.array(templatePoints)
        p1 = np.array(imagePoints)
        
        _L, model = self.approximate(p0, p1)
        
        # For all coordinates in the register data, evaluate the
        # thin-plate-spline.
        
        warp = np.zeros_like(image.coords.tensor)
        
        affine  = model[-3:, :]
        weights = model[:-3, :]
        
        if vectorized:
            
            # Vectorized extrapolation, looping through arrays in python is 
            # slow therefore wherever possible attempt to unroll loops. Below
            # is an example:
            
            Xvec = np.matrix(image.coords.tensor[1].flatten(0)).T
            Yvec = np.matrix(image.coords.tensor[0].flatten(0)).T
       
            # Fast matrix multiplication approach:
            Px = np.matrix(np.tile(p0[:,0], (Xvec.shape[0], 1)))
            Wx = np.matrix(np.tile(weights[:,0].T, (Xvec.shape[0], 1)))
            Bx = np.matrix(np.tile(Xvec, (1, p0.shape[0])))
            Ax = np.matrix(np.tile(affine[:,0].T, (Xvec.shape[0], 1)))
        
            Py = np.matrix(np.tile(p0[:,1], (Xvec.shape[0], 1)))
            Wy = np.matrix(np.tile(weights[:,1].T, (Xvec.shape[0], 1)))
            By = np.matrix(np.tile(Yvec, (1, p0.shape[0])))
            Ay = np.matrix(np.tile(affine[:,1].T, (Xvec.shape[0], 1)))
        
            # Form the R matrix:
            R = self.U( np.sqrt( np.power(Px - Bx,2) + np.power(Py - By,2)) )
        
            # Compute the sum of the weighted R matrix, row wise.
            Rx = np.sum( np.multiply(Wx, R), 1 )
            Ry = np.sum( np.multiply(Wy, R), 1 )
        
            one = np.ones_like(Xvec)
            a = np.hstack(( one, Xvec, Yvec, one))
   
            warp[1] = np.sum( np.multiply(a, np.hstack((Ax, Rx)) ), 1 ).reshape(warp[1].shape)
            warp[0] = np.sum( np.multiply(a, np.hstack((Ay, Ry)) ), 1 ).reshape(warp[0].shape)
            
        else:
            
            # Slow nested loop approach:
            for x in xrange(0, image.coords.domain[1]):
                for y in xrange(0, image.coords.domain[3]):
                    
                    # Refer to page 570 (of BookStein paper) first column, last 
                    # equation (relating map coordinates to coefficients).
                    
                    zx = 0.0
                    zy = 0.0
                    
                    for n in range(0, len(p0[:,0])):
                        r = np.sqrt( (p0[n,0] - x)**2 + (p0[n,1] - y)**2 ) 
                        zx += float(weights[n,0])*float(self.U(r)) 
                        zy += float(weights[n,1])*float(self.U(r)) 
            
                    warp[0][y,x] = affine[0,1] + affine[1,1]*x + affine[2,1]*y + zy
                    warp[1][y,x] = affine[0,0] + affine[1,0]*x + affine[2,0]*y + zx
        
        img = sampler.f(image.data, warp).reshape(image.data.shape)
        
        return warp, img
