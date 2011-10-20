""" A collection of image similarity metrics. """
from ssim import calc_ssim, imageshash, flush_work
import numpy as np
import ssim



class Metric(object):
    """
    Abstract similarity metric.
    
    Attributes
    ----------
    METRIC : string
        The type of similarity metric being used.
    DESCRIPTION : string
        A meaningful description of the metric used, with references where 
        appropriate.
    """

    METRIC=None
    DESCRIPTION=None

    def __init__(self):
        pass

    def error(self, warpedImage, template):
        """
        Evaluates the metric.
        
        Parameters
        ----------
        warpedImage: nd-array
            Input image after warping.
        template: nd-array
            Template image.
    
        Returns
        -------
        error: nd-array
           Metric evaluated over all image coordinates.
        """
        
        raise NotImplementedError('')
        
    def jacobian(self, model, warpedImage, p=None):
        """
        Computes the jacobian dP/dE.
        
        Parameters
        ----------
        model: deformation model
            A particular deformation model.
        warpedImage: nd-array
            Input image after warping.
        p : optional list
            Current warp parameters
    
        Returns
        -------
        jacobian: nd-array
           A derivative of model parameters with respect to the metric.
        """
        raise NotImplementedError('')
        
    def __str__(self):
        return 'Metric: {0} \n {1}'.format(
            self.METRIC,
            self.DESCRIPTION
            )


class Residual(Metric):
    """ Standard least squares metric """

    METRIC='residual'

    DESCRIPTION="""
        The residual which is computed as the difference between the
        deformed image an the template:

            (I(W(x;p)) - T)

        """

    def __init__(self):
        Metric.__init__(self)

    def jacobian(self, model, warpedImage, p=None):
        """
        Computes the jacobian dP/dE.
        
        Parameters
        ----------
        model: deformation model
            A particular deformation model.
        warpedImage: nd-array
            Input image after warping.
        p : optional list
            Current warp parameters
    
        Returns
        -------
        jacobian: nd-array
            A jacobain matrix. (m x n)
                | where: m = number of image pixels,
                |        p = number of parameters.
        """

        grad = np.gradient(warpedImage)

        dIx = grad[1].flatten()
        dIy = grad[0].flatten()

        dPx, dPy = model.jacobian(p)

        J = np.zeros_like(dPx)
        for index in range(0, dPx.shape[1]):
            J[:,index] = dPx[:,index]*dIx + dPy[:,index]*dIy
        return J

    def error(self, warpedImage, template):
        """
        Evaluates the residual metric.
        
        Parameters
        ----------
        warpedImage: nd-array
            Input image after warping.
        template: nd-array
            Template image.
    
        Returns
        -------
        error: nd-array
           Metric evaluated over all image coordinates.
        """
        return warpedImage.flatten() - template.flatten()



def _dssim(image, reference, window):
    hashkey = imageshash(image,reference)
    flush_work()
    dssim = np.zeros(image.shape)
    for j in range(image.shape[0]):
        # print 100.0 * j / image.shape[0]
        for i in range(image.shape[1]):
            bbox = [max(0,i-window/2), max(0,j-window/2), min(image.shape[1], i+window/2), min(image.shape[0], j+window/2)]
            dssim[j,i] = calc_ssim(image, reference, bbox, direct=False, imagekey=hashkey)
    flush_work()        
    return dssim


class Dssim(Metric):
    """ DSSIM metric """

    METRIC='dssim'

    DESCRIPTION="""
        The error metric which is computed as the DSSIM between the
        deformed image and the template:

            DSSIM(I(W(x;p)),T)

        """

    def __init__(self):
        Metric.__init__(self)

    def jacobian(self, model, warpedImage, p=None):
        """
        Computes the jacobian dP/dE.
        
        Parameters
        ----------
        model: deformation model
            A particular deformation model.
        warpedImage: nd-array
            Input image after warping.
        p : optional list
            Current warp parameters
    
        Returns
        -------
        jacobian: nd-array
            A jacobain matrix. (m x n)
                | where: m = number of image pixels,
                |        p = number of parameters.
        """

        grad = np.gradient(warpedImage)

        dIx = grad[1].flatten()
        dIy = grad[0].flatten()

        dPx, dPy = model.jacobian(p)

        J = np.zeros_like(dPx)
        for index in range(0, dPx.shape[1]):
            J[:,index] = dPx[:,index]*dIx + dPy[:,index]*dIy
        return J

    def error(self, warpedImage, template):
        """
        Evaluates the DSSIM metric.
        
        Parameters
        ----------
        warpedImage: nd-array
            Input image after warping.
        template: nd-array
            Template image.
    
        Returns
        -------
        error: nd-array
           Metric evaluated over all image coordinates.
        """
        d = _dssim(warpedImage, template, 11)
        return d.flatten()
