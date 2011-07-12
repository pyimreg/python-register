""" A collection of image similarity metrics. """

import numpy as np

class metric(object):
    """
    Abstract similarity metric.
    
    @param METRIC: the type of similarity metric being used.
    @param DESCRIPTION: a meaningful description of the metric used, with 
                        references where appropriate. 
    """
    
    METRIC=None
    DESCRIPTION=None
    
    def __init__(self):
        pass
    
    def error(self, warpedImage, template):
        """
        Computes the metric.
        
        @param warpedImage: a numpy array, representing the image.
        @param template: a numpy arrary, representing the template. 
        """
        return None
    
    def jacobian(self, model, warpedImage):
        """
        Computes the jacobian dP/dE
        
        @param model: the deformation model.
        @param warpedImage: the transformed image.
        """
        return None
        
    def __str__(self):
        return 'Metric: {0} \n {1}'.format(
            self.METRIC,
            self.DESCRIPTION
            )


class residual(metric):
    """ Standard least squares metric """
    
    METRIC='residual'
    
    DESCRIPTION="""
        The residual which is computed as the difference between the
        deformed image an the template:
        
            (I(W(x;p)) - T)
        
        """
    
    def __init__(self):
        metric.__init__(self)
    
    def jacobian(self, model, warpedImage):
        """
        Follows the derivations shown in:
        
        Simon Baker and Iain Matthews. 2004. Lucas-Kanade 20 Years On: A 
        Unifying Framework. Int. J. Comput. Vision 56, 3 (February 2004).
        
        @param model: the deformation model. 
        @param warpedModel: a transformed image.
        @return: a jacobain matrix. (m x n) 
            where: m = number of image pixels, 
                   p = number of parameters.
        """
        
        grad = np.gradient(warpedImage)
        
        dIx = grad[1].flatten()
        
        dIy = grad[0].flatten()
        
        dPx, dPy = model.jacobian()
        
        J = np.zeros_like(dPx)
        for index in range(0, dPx.shape[1]):
            J[:,index] = dPx[:,index]*dIx + dPy[:,index]*dIy
        
        return J
    
    def error(self, warpedImage, template):
        
        return warpedImage.flatten() - template.flatten()
    