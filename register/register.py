""" A top level registration module """

import collections
import numpy as np
import scipy.ndimage as nd


def _smooth(image, variance):
    """
    Gaussian smoothing using the fast-fourier-transform (FFT)
    
    Parameters
    ----------
    image: nd-array
        Input image 
    variance: float
        Variance of the Gaussian kernel.

    Returns
    -------
    image: nd-array
       An image convolved with the Gaussian kernel.

    See also
    --------
    regisger.Register.smooth
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
    Container for grid coordinates.
    
    Attributes
    ----------
    domain : nd-array
        Domain of the coordinate system.
    tensor : nd-array
        Grid coordinates.
    homogenous : nd-array
        `Homogenous` coordinate system representation of grid coordinates.
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
    Container for registration data.
    
    Attributes
    ----------
    data : nd-array
        The image registration image values.
    coords : nd-array, optional
        The grid coordinates.
    features : dictionary, optional
        A mapping of unique ids to registration features.
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
        # labeled salient image coordinates (point features). 
        
        self.features = features
        
        
    def smooth(self, variance):
        """
        Smooth feature data in place.
    
        Parameters
        ----------
        variance: float
            Variance of the Gaussian kernel.
       
        See also
        --------
        regisger.Register.smooth
        """

        self.data = _smooth(self.data, variance)


class Register(object):
    """
    A registration class for estimating the deformation model parameters that
    best solve:
    
    | :math:`f( W(I;p), T )`
    |
    | where:
    |    :math:`f`     : is a similarity metric.
    |    :math:`W(x;p)`: is a deformation model (defined by the parameter set p).
    |    :math:`I`     : is an input image (to be deformed).
    |    :math:`T`     : is a template (which is a deformed version of the input).
    
    Notes:
    ------
    
    Solved using a modified gradient descent algorithm.
    
    .. [0] Levernberg-Marquardt algorithm, 
           http://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm 
    
    Attributes
    ----------
    model: class
        A `deformation` model class definition.
    metric: class
        A `similarity` metric class definition.
    sampler: class
        A `sampler` class definition.
    """
    
    optStep = collections.namedtuple('optStep', 'error p deltaP')
    
    MAX_ITER = 200
    MAX_BAD = 20
    
    def __init__(self, model, metric, sampler):

        self.model = model
        self.metric = metric
        self.sampler = sampler

    def __deltaP(self, J, e, alpha, p=None):
        """
        Computes the parameter update.
        
        Parameters
        ----------
        J: nd-array
            The (dE/dP) the relationship between image differences and model
            parameters.
        e: float
            The evaluated similarity metric.    
        alpha: float
            A dampening factor.
        p: nd-array or list of floats, optional
        
        Returns
        -------
        deltaP: nd-array
           The parameter update vector.
        """

        H = np.dot(J.T, J)

        H += np.diag(alpha*np.diagonal(H))

        return np.dot( np.linalg.inv(H), np.dot(J.T, e))

    def __dampening(self, alpha, decreasing):
        """
        Computes the adjusted dampening factor.
        
        Parameters
        ----------
        alpha: float
            The current dampening factor.
        decreasing: boolean
            Conditional on the decreasing error function.
            
        Returns
        -------
        alpha: float
           The adjusted dampening factor.
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
        Computes the registration between the image and template.
        
        Parameters
        ----------
        image: nd-array
            The floating image.
        template: nd-array
            The target image.
        p: list (or nd-array), optional.
            First guess at fitting parameters.
        warp: nd-array, optional.
            A warp field estimate.
        alpha: float
            The dampening factor.
        plotCB: function, optional
            A plotting function.
        verbose: boolean
            A debug flag for text status updates. 
        
        Returns
        -------
        p: nd-array.
            Model parameters.
        warp: nd-array.
            Warp field estimate.
        warpedImage: nd-array
            The re-sampled image.
        error: float
            Fitting error.
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
        Computes the parameter update.
        
        Parameters
        ----------
        J: nd-array
            The (dE/dP) the relationship between image differences and model
            parameters.
        e: float
            The evaluated similarity metric.    
        alpha: float
            A dampening factor.
        p: nd-array or list of floats, optional
        
        Returns
        -------
        deltaP: nd-array
           The parameter update vector.
        """

        H = np.dot(J.T, J)

        H += np.diag(alpha*np.diagonal(H))

        return np.dot( np.linalg.inv(H), np.dot(J.T, e)) - alpha*p

    def __dampening(self, alpha, decreasing):
        """
        Computes the adjusted dampening factor.
        
        Parameters
        ----------
        alpha: float
            The current dampening factor.
        decreasing: boolean
            Conditional on the decreasing error function.
            
        Returns
        -------
        alpha: float
           The adjusted dampening factor.
        """
        return alpha


class FeatureRegister():
    """
    A registration class for estimating the deformation model parameters that
    best solve:
    
    | :math:`\arg\min_{p} | f(p_0) - p_1 |`
    |
    | where:
    |    :math:`f`     : is a transformation function.
    |    :math:`p_0`   : is a deformation model (defined by the parameter set p).
    |    :math:`p_1`   : is an input image (to be deformed).
     
    Notes:
    ------
    
    Solved using linear algebra - does not consider pixel intensities
    
    Attributes
    ----------
    model: class
        A `deformation` model class definition.
    sampler: class
        A `sampler` class definition.
    """
    
    def __init__(self, model, sampler):

        self.model = model
        self.sampler = sampler

    def register(self, image, template):
        """
        Computes the registration using only image (point) features.
            
        Parameters
        ----------
        image: RegisterData
            The floating registration data.
        template: RegisterData
            The target registration data.
        model: Model
            The deformation model.
            
        Returns
        -------
        p: nd-array.
                Model parameters.
        warp: nd-array.
            Warp field estimate.
        warpedImage: nd-array
            The re-sampled image.
        error: float
            Fitting error.
        """
        
        # Initialize the models, metric and sampler.
        model = self.model(image.coords)
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
            raise ValueError('Requires corresponding features to register.')
        
        # Note the inverse warp is estimated here.
        p = model.fit(
            np.array(templatePoints),
            np.array(imagePoints)
            )
        
        warp = model.warp(p)
        
        # Sample the image using the inverse warp.
        warpedImage = sampler.f(image.data, warp).reshape(image.data.shape)
        
        return p, warp, warpedImage
