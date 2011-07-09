""" A collection of deformation models. """

import numpy as np
import scipy.signal as signal

class model(object):
    """
    Abstract geometry model.
    
    @param MODEL: the deformation model used.
    @param DESCRIPTION: a meaningful description of the model used, with 
                        references where appropriate. 
    """
    
    MODEL=None
    DESCRIPTION=None
    
    def __init__(self, coordinates):
        
        self.coordinates = coordinates
    
    def estimate(self, warp):
        """"
        Estimates the best fit parameters that define a warp field.
        
        @param warp: a warp field, representing the warped coordinates. 
        @return: a set of parameters (n-dimensional array).
        """
        
        return self.identity
        
    def warp(self, parameters):
        """
        Computes the warp field given transformed coordinates.
       
        @param param: array coordinates of model parameters.
        @return: a deformation field.
        """
        
        coords = self.transform(parameters)
        
        warp = np.zeros_like(self.coordinates.grid)
        
        warp[0] = coords[1].reshape(warp[0].shape)
        warp[1] = coords[0].reshape(warp[1].shape)
        
        # Return the difference warp grid.
        return warp
        
    def transform(self, parameters):
        """
        A geometric transformation of coordinates.
       
        @param param: array coordinates of model parameters.
        """
        
        return None
        
    def __str__(self):
        return 'Model: {0} \n {1}'.format(
            self.MODEL,
            self.DESCRIPTION
            )


class shift(model):
    
    MODEL='Shift (S)'
    
    DESCRIPTION="""
        Applies the shift coordinate transformation.        
                """
    
    def __init__(self, coordinates):
        model.__init__(self, coordinates)
    
    @property
    def identity(self):
        return np.array([0.0]*2)
    
    def transform(self, parameters):
        """
        
        Applies an shift transformation to image coordinates.
        
        @param parameters: a array of shift parameters.
        @param coords: array coordinates in cartesian form (n by p).
        @return: a transformed set of coordinates.
        """
        
        T = np.eye(3,3)
        T[0,2] = -parameters[0]
        T[1,2] = -parameters[1]
        
        return np.dot(T, self.coordinates.homogenous)
    
    def jacobian(self):
        """
        Follows the derivations shown in:
        
        Simon Baker and Iain Matthews. 2004. Lucas-Kanade 20 Years On: A 
        Unifying Framework. Int. J. Comput. Vision 56, 3 (February 2004).
         
        Evaluates the derivative of deformation model with respect to the 
        coordinates.
        """
        
        dx = np.zeros((self.coordinates.grid[0].size, 2))
        dy = np.zeros((self.coordinates.grid[0].size, 2))
        
        dx[:,0] = 1
        dy[:,1] = 1
        
        return (dx, dy)


class affine(model):
    
    MODEL='Affine (A)'
    
    DESCRIPTION="""
        Applies the affine coordinate transformation.        
                """
    
    def __init__(self, coordinates):
        model.__init__(self, coordinates)
    
    @property
    def identity(self):
        return np.array([0.0]*6)
    
    def transform(self, p):
        """
        Applies an affine transformation to image coordinates.
        
        @param parameters: a array of affine parameters.
        @param coords: array coordinates in cartesian form (n by p).
        @return: a transformed set of coordinates.
        """
        
        T = np.array([
                      [p[0]+1.0, p[2],     p[4]],
                      [p[1],     p[3]+1.0, p[5]],
                      [0,         0,         1]
                      ])
        
        return np.dot(np.linalg.inv(T), self.coordinates.homogenous)
    
    def jacobian(self):
        """
        Follows the derivations shown in:
        
        Simon Baker and Iain Matthews. 2004. Lucas-Kanade 20 Years On: A 
        Unifying Framework. Int. J. Comput. Vision 56, 3 (February 2004).
         
        Evaluates the derivative of deformation model with respect to the 
        coordinates.
        """
        
        dx = np.zeros((self.coordinates.grid[0].size, 6))
        dy = np.zeros((self.coordinates.grid[0].size, 6))
        
        dx[:,0] = self.coordinates.grid[1].flatten()
        dx[:,2] = self.coordinates.grid[0].flatten()
        dx[:,4] = 1.0
        
        dy[:,1] = self.coordinates.grid[1].flatten()
        dy[:,3] = self.coordinates.grid[0].flatten()
        dy[:,5] = 1.0
        
        return (dx, dy)


class spline(model):
    
    MODEL='Spline (S)'
    
    DESCRIPTION="""
        Applies a spline deformation model, as described in:
        
        Kybic, J. and Unser, M. (2003). Fast parametric elastic image 
        registration. IEEE Transactions on Image Processing, 12(11), 1427-1442. 
        """
    
    def __init__(self, coordinates):
        
        model.__init__(self, coordinates)
        self.__basis()
    
    @property
    def identity(self):
        return np.array([0.0]*(self.basis.shape[1]*2))
    
    @property
    def numberOfParameters(self):
        return self.basis.shape[1]
    
    def __basis(self, order=4, divisions=4):
        """
        Follows the derivations in:
        
        Kybic, J. and Unser, M. (2003). Fast parametric elastic image 
        registration. IEEE Transactions on Image Processing, 12(11), 1427-1442. 
        
        Computes the spline tensor product and stores the products, as basis
        vectors.
        
        @param order: b-spline order.
        @param division: number of spline knots. 
        """
        
        shape = self.coordinates.grid[0].shape
        grid = self.coordinates.grid
        
        spacing = shape[1] / divisions
        xKnots = shape[1] / spacing
        yKnots = shape[0] / spacing
        
        Qx = np.zeros((grid[0].size, xKnots))
        Qy = np.zeros((grid[0].size, yKnots))
        
        for index in range(0, xKnots):
            bx = signal.bspline( grid[1] / spacing - index, order)
            Qx[:,index] = bx.flatten()
           
        for index in range(0, yKnots):
            by = signal.bspline( grid[0] / spacing - index, order)
            Qy[:,index] = by.flatten() 
           
        basis = []
        for j in range(0,xKnots):
            for k in range(0, yKnots):
                basis.append(Qx[:,j]*Qy[:,k])
        
        self.basis = np.array(basis).T
        
    
    def estimate(self, warp):
        """"
        Estimates the best fit parameters that define a warp field.
        
        @param warp: a warp field, representing the warped coordinates. 
        @return: a set of parameters (n-dimensional array).
        """
        
        return np.hstack( 
            (
             np.dot(np.linalg.pinv(self.basis),
                    (self.coordinates.grid[0] - warp[0]).flatten()),
             np.dot(np.linalg.pinv(self.basis),
                    (self.coordinates.grid[1] - warp[1]).flatten()),
            )
           ).T
        
        
    def warp(self, parameters):
        """
        Computes the (inverse) warp field given transformed coordinates.
       
        @param param: array coordinates of model parameters.
        @return: a deformation field.
        """
        
        dwarp = self.transform(parameters)
        return self.coordinates.grid - dwarp
    
    def transform(self, p):
        """
        Applies an spline transformation to image coordinates.
        
        @param parameters: a array of affine parameters.
        @param coords: array coordinates in cartesian form (n by p).
        @return: a transformed set of coordinates.
        """
        
        px = np.array(p[0:self.numberOfParameters])
        py = np.array(p[self.numberOfParameters::])
        
        shape = self.coordinates.grid[0].shape
        
        return np.array( [ np.dot(self.basis, py).reshape(shape), 
                           np.dot(self.basis, px).reshape(shape)
                         ]
                       )
        
    def jacobian(self):
        """
        Follows the derivations shown in:
        
        Kybic, J., & Unser, M. (2003). Fast parametric elastic image 
        registration. IEEE Transactions on Image Processing, 12(11), 1427-1442. 
       
        Evaluate the derivative of deformation model with respect to the 
        coordinates.
        """
        
        dx = np.zeros((self.coordinates.grid[0].size, 
                       2*self.numberOfParameters))
        
        dy = np.zeros((self.coordinates.grid[0].size, 
                       2*self.numberOfParameters))
        
        dx[:, 0:self.numberOfParameters] = self.basis
        dy[:, self.numberOfParameters::] = self.basis
                
        return (dx, dy)
