""" A collection of image similarity metrics. """

import numpy as np

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
            J[:, index] = dPx[:, index] * dIx + dPy[:, index] * dIy
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
