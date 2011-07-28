""" A top level registration module """

import collections
import numpy as np
import scipy.ndimage as nd


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
    def __init__(self, data, coords=None):

        self.data = data
        
        if not coords:
            self.coords = Coordinates(
                [0, data.shape[0], 0, data.shape[1]]
                )
        else:
            self.coords = coords
        
        self.features = None
        
        
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

        @param p: a list of parameters, (first guess).
        @param floating: a floating image, numpy ndarray.
        @param target: a target image, numpy ndarray.
        @keyword mask: an optional parameter mask.
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
    TBD
    """
    
    def U(self, r):
        
        # Thin plate spline kernel
        # 
        return np.multiply( -np.power(r,2), np.log(np.power(r,2) + 1e-20))
        
        # Gaussian kernel
        
        #var = 15
        #return np.exp( -pow(r,2)/(2*var**2)  )
    
    def __approximate(self, p0, p1):
        
        K = np.zeros((p0.shape[0], p0.shape[0]))
        
        for i in range(0, p0.shape[0]):
            for j in range(0, p0.shape[0]):
                r = sqrt( (p0[i,0] - p0[j,0])**2 + (p0[i,1] - p0[j,1])**2 ) 
                K[i,j] = self.U(r)
                
        P = np.hstack((np.ones((p0.shape[0], 1)), p0))
        
        L = np.vstack(
                      ( 
                       np.hstack((K,P)), 
                       np.hstack((P.transpose(), np.zeros((3,3)))) 
                      )
                     )
        
        Y = np.vstack( (p1, np.zeros((3, 2))) )
        
        Y = np.matrix(Y)
        
        Linv = np.matrix(np.linalg.inv(L))
        
        return ( Linv*Y)    
    
    @print_timing
    def fit(self, p0, p1):
        
        X, Y = self.fastfit(p0, p1)
        
        model = self.__approximate(p0, p1)
        
        affine  = model[-3:, :]
        weights = model[:-3, :]
        
        X = np.zeros(self.domain)
        Y = np.zeros(self.domain)
        
        # Form the basis vectors
        
        for x in self.xRange:
            for y in self.yRange:
                
                zx = 0.0
                zy = 0.0
                
                for n in range(0, len(p0[:,0])):
                    r = sqrt( (p0[n,0] - x)**2 + (p0[n,1] - y)**2 ) 
                    zx += float(weights[n,0])*float(self.U(r)) 
                    zy += float(weights[n,1])*float(self.U(r)) 
            
                X[y,x] = affine[0,0] + affine[1,0]*x + affine[2,0]*y + zx
                Y[y,x] = affine[0,1] + affine[1,1]*x + affine[2,1]*y + zy
        
        return (X,Y)
    
    @print_timing
    def fastfit(self, p0, p1):
        
        model = self.__approximate(p0, p1)
        
        affine  = model[-3:, :]
        weights = model[:-3, :]
        
        # Form the basis vectors
        
        X, Y = np.meshgrid(self.xRange, self.yRange)
        
        Xvec = np.matrix(X.flatten(0)).T
        Yvec = np.matrix(Y.flatten(0)).T
        
        Px = np.matrix(np.tile(p0[:,0], (Xvec.shape[0], 1)))
        Wx = np.matrix(np.tile(weights[:,0].T, (Xvec.shape[0], 1)))
        Bx = np.matrix(np.tile(Xvec, (1, p0.shape[0])))
        Ax = np.matrix(np.tile(affine[:,0].T, (Xvec.shape[0], 1)))
        
        Py = np.matrix(np.tile(p0[:,1], (Xvec.shape[0], 1)))
        Wy = np.matrix(np.tile(weights[:,1].T, (Xvec.shape[0], 1)))
        By = np.matrix(np.tile(Yvec, (1, p0.shape[0])))
        Ay = np.matrix(np.tile(affine[:,1].T, (Xvec.shape[0], 1)))
        
        # Form the R matrix
        R = self.U( np.sqrt( np.power(Px - Bx,2) + np.power(Py - By,2)) )
        
        # Compute the sum of the weighted R matrix, row wise.
        Rx = np.sum( np.multiply(Wx, R), 1 )
        Ry = np.sum( np.multiply(Wy, R), 1 )
        
        one = np.ones_like(Xvec)
        
        a = np.hstack(( one, Xvec, Yvec, one))
   
        Nx = np.sum( np.multiply(a, np.hstack((Ax, Rx)) ), 1 )
        Ny = np.sum( np.multiply(a, np.hstack((Ay, Ry)) ), 1 )
        
        # Debug visualization of the displacement grid
        
        plt.figure()
        ax=plt.subplot(121)
        ax.matshow(np.reshape(np.abs(Xvec-Nx), self.domain))
        ax=plt.subplot(122)
        ax.matshow(np.reshape(np.abs(Yvec-Ny), self.domain))
        
        
        return( np.reshape(Nx, self.domain), np.reshape(Ny, self.domain))
    
    
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

        @param p: a list of parameters, (first guess).
        @param floating: a floating image, numpy ndarray.
        @param target: a target image, numpy ndarray.
        @keyword mask: an optional parameter mask.
        """
    
    
    
    
    
    
    
        
    
    