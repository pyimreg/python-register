""" A collection of deformation models. """

import numpy as np
import scipy.signal as signal


class Model(object):
    """
    Abstract geometry model.
    
    Attributes
    ----------
    METRIC : string
        The type of similarity metric being used.
    DESCRIPTION : string
        A meaningful description of the model used, with references where 
        appropriate.
    """

    MODEL=None
    DESCRIPTION=None

    def __init__(self, coordinates):
        self.coordinates = coordinates
    
    def fit(self, p0, p1):
        """
        Estimates the best fit parameters that define a warp field, which 
        deforms feature points p0 to p1.
        
        Parameters
        ----------
        p0: nd-array
            Image features (points).
        p1: nd-array
            Template features (points).
            
        Returns
        -------
        parameters: nd-array
            Model parameters.
        error: float
            Sum of RMS error between p1 and alinged p0.
        """
        raise NotImplementedError('')
    
    @staticmethod
    def scale(p, factor):
        """
        Scales an transformtaion by a factor.
        
        Parameters
        ----------
        p: nd-array
            Model parameters.
        factor: float
            A scaling factor.
            
        Returns
        -------
        parameters: nd-array
            Model parameters.
        """
        raise NotImplementedError('')
    
    
    def estimate(self, warp):
        """
        Estimates the best fit parameters that define a warp field.
        
        Parameters
        ----------
        warp: nd-array
            Deformation field.
        
        Returns
        -------
        parameters: nd-array
           Model parameters.
        """
        raise NotImplementedError('')
        
    def warp(self, parameters):
        """
        Computes the warp field given model parameters.
        
        Parameters
        ----------
        parameters: nd-array
            Model parameters.
        
        Returns
        -------
        warp: nd-array
           Deformation field.
        """
        
        coords = self.transform(parameters)

        warp = np.zeros_like(self.coordinates.tensor)

        warp[0] = coords[1].reshape(warp[0].shape)
        warp[1] = coords[0].reshape(warp[1].shape)

        # Return the difference warp grid.
        return warp

    def transform(self, parameters):
        """
        A geometric transformation of coordinates.
        
        Parameters
        ----------
        parameters: nd-array
            Model parameters.
        
        Returns
        -------
        coords: nd-array
           Deformation coordinates.
        """
        raise NotImplementedError('')
    
    def jacobian(self, p=None):
        """
        Evaluates the derivative of deformation model with respect to the
        coordinates.
        """
        raise NotImplementedError('')
        
    def __str__(self):
        return 'Model: {0} \n {1}'.format(
            self.MODEL,
            self.DESCRIPTION
            )


class Shift(Model):

    MODEL='Shift (S)'

    DESCRIPTION="""
        Applies the shift coordinate transformation. Follows the derivations 
        shown in:
      
        S. Baker and I. Matthews. 2004. Lucas-Kanade 20 Years On: A
        Unifying Framework. Int. J. Comput. Vision 56, 3 (February 2004).
        """

    def __init__(self, coordinates):
        Model.__init__(self, coordinates)

    @property
    def identity(self):
        return np.zeros(2)
    
    @staticmethod
    def scale(p, factor):
        """
        Scales an shift transformation by a factor.
        
        Parameters
        ----------
        p: nd-array
            Model parameters.
        factor: float
            A scaling factor.
            
        Returns
        -------
        parameters: nd-array
            Model parameters.
        """
        
        pHat = p.copy()
        pHat *= factor
        return pHat
    
    
    def fit(self, p0, p1, lmatrix=False):
        """
        Estimates the best fit parameters that define a warp field, which 
        deforms feature points p0 to p1.
        
        Parameters
        ----------
        p0: nd-array
            Image features (points).
        p1: nd-array
            Template features (points).
            
        Returns
        -------
        parameters: nd-array
            Model parameters.
        error: float
            Sum of RMS error between p1 and alinged p0.
        """
        
        parameters = p1.mean(axis=0) - p0.mean(axis=0)
        
        projP0 = p0 + parameters
        
        error = np.sqrt( 
           (projP0[:,0] - p1[:,0])**2 + (projP0[:,1] - p1[:,1])**2 
           ).sum() 
        
        return -parameters, error
    
    def transform(self, parameters):
        """
        A "shift" transformation of coordinates.
        
        Parameters
        ----------
        parameters: nd-array
            Model parameters.
        
        Returns
        -------
        coords: nd-array
           Deformation coordinates.
        """
        
        T = np.eye(3,3)
        T[0,2] = -parameters[0]
        T[1,2] = -parameters[1]

        return np.dot(T, self.coordinates.homogenous)

    def jacobian(self, p=None):
        """
        Evaluates the derivative of deformation model with respect to the
        coordinates.
        """

        dx = np.zeros((self.coordinates.tensor[0].size, 2))
        dy = np.zeros((self.coordinates.tensor[0].size, 2))

        dx[:,0] = 1
        dy[:,1] = 1

        return (dx, dy)


class Affine(Model):

    MODEL='Affine (A)'

    DESCRIPTION="""
        Applies the affine coordinate transformation. Follows the derivations 
        shown in:
        
        S. Baker and I. Matthews. 2004. Lucas-Kanade 20 Years On: A
        Unifying Framework. Int. J. Comput. Vision 56, 3 (February 2004).
        """

    def __init__(self, coordinates):
        Model.__init__(self, coordinates)

    @property
    def identity(self):
        return np.zeros(6)
    
    @staticmethod
    def scale(p, factor):
        """
        Scales an affine transformation by a factor.
        
        Parameters
        ----------
        p: nd-array
            Model parameters.
        factor: float
            A scaling factor.
            
        Returns
        -------
        parameters: nd-array
            Model parameters.
        """
        
        pHat = p.copy()
        pHat[4:] *= factor
        return pHat
    
    def fit(self, p0, p1, lmatrix=False):
        """
        Estimates the best fit parameters that define a warp field, which 
        deforms feature points p0 to p1.
        
        Parameters
        ----------
        p0: nd-array
            Image features (points).
        p1: nd-array
            Template features (points).
            
        Returns
        -------
        parameters: nd-array
            Model parameters.
        error: float
            Sum of RMS error between p1 and alinged p0.
        """
        
        # Solve: H*X = Y
        # ---------------------
        #          H = Y*inv(X)
        
        X = np.ones((3, len(p0)))
        X[0:2,:] = p0.T
        
        Y = np.ones((3, len(p0)))
        Y[0:2,:] = p1.T
        
        H = np.dot(Y, np.linalg.pinv(X))
        
        parameters = [ 
            H[0,0] - 1.0,
            H[1,0],
            H[0,1],
            H[1,1] - 1.0,
            H[0,1],
            H[1,2]
            ]
        
        projP0 = np.dot(H, X)[0:2,:].T
        
        error = np.sqrt( 
           (projP0[:,0] - p1[:,0])**2 + (projP0[:,1] - p1[:,1])**2 
           ).sum() 
        
        return parameters, error
        
    def transform(self, p):
        """
        An "affine" transformation of coordinates.
        
        Parameters
        ----------
        parameters: nd-array
            Model parameters.
        
        Returns
        -------
        coords: nd-array
           Deformation coordinates.
        """
        
        T = np.array([
                      [p[0]+1.0, p[2],     p[4]],
                      [p[1],     p[3]+1.0, p[5]],
                      [0,         0,         1]
                      ])

        return np.dot(np.linalg.inv(T), self.coordinates.homogenous)

    def jacobian(self, p=None):
        """"
        Evaluates the derivative of deformation model with respect to the
        coordinates.
        """

        dx = np.zeros((self.coordinates.tensor[0].size, 6))
        dy = np.zeros((self.coordinates.tensor[0].size, 6))

        dx[:,0] = self.coordinates.tensor[1].flatten()
        dx[:,2] = self.coordinates.tensor[0].flatten()
        dx[:,4] = 1.0

        dy[:,1] = self.coordinates.tensor[1].flatten()
        dy[:,3] = self.coordinates.tensor[0].flatten()
        dy[:,5] = 1.0

        return (dx, dy)

class Projective(Model):

    MODEL='Projective (P)'

    DESCRIPTION="""
        Applies the projective coordinate transformation. Follows the derivations 
        shown in:
        
        S. Baker and I. Matthews. 2004. Lucas-Kanade 20 Years On: A
        Unifying Framework. Int. J. Comput. Vision 56, 3 (February 2004).
        """

    def __init__(self, coordinates):
        Model.__init__(self, coordinates)

    @property
    def identity(self):
        return np.zeros(9)
    
    def fit(self, p0, p1, lmatrix=False):
        """
        Estimates the best fit parameters that define a warp field, which 
        deforms feature points p0 to p1.
        
        Parameters
        ----------
        p0: nd-array
            Image features (points).
        p1: nd-array
            Template features (points).
            
        Returns
        -------
        parameters: nd-array
            Model parameters.
        error: float
            Sum of RMS error between p1 and alinged p0.
        """
        
        # Solve: H*X = Y
        # ---------------------
        #          H = Y*inv(X)
        
        X = np.ones((3, len(p0)))
        X[0:2,:] = p0.T
        
        Y = np.ones((3, len(p0)))
        Y[0:2,:] = p1.T
        
        H = np.dot(Y, np.linalg.pinv(X))
        
        parameters = [ 
            H[0,0] - 1.0,
            H[1,0],
            H[0,1],
            H[1,1] - 1.0,
            H[0,2],
            H[1,2],
            H[2,0],
            H[2,1],
            H[2,2] - 1.0
            ]
        
        projP0 = np.dot(H, X)[0:2,:].T
        
        error = np.sqrt( 
           (projP0[:,0] - p1[:,0])**2 + (projP0[:,1] - p1[:,1])**2 
           ).sum() 
        
        return parameters, error
        
    def transform(self, p):
        """
        An "projective" transformation of coordinates.
        
        Parameters
        ----------
        parameters: nd-array
            Model parameters.
        
        Returns
        -------
        coords: nd-array
           Deformation coordinates.
        """
        
        T = np.array([
                      [p[0]+1.0, p[2],     p[4]],
                      [p[1],     p[3]+1.0, p[5]],
                      [p[6],     p[7],     p[8]+1.0]
                      ])

        return np.dot(np.linalg.inv(T), self.coordinates.homogenous)

    def jacobian(self, p):
        """"
        Evaluates the derivative of deformation model with respect to the
        coordinates.
        """

        dx = np.zeros((self.coordinates.tensor[0].size, 9))
        dy = np.zeros((self.coordinates.tensor[0].size, 9))
    
        x = self.coordinates.tensor[1].flatten()
        y = self.coordinates.tensor[0].flatten()
        
        dx[:,0] = x / (p[6]*x + p[7]*y + p[8] + 1) 
        dx[:,2] = y / (p[6]*x + p[7]*y + p[8] + 1) 
        dx[:,4] = 1.0 / (p[6]*x + p[7]*y + p[8] + 1) 
        dx[:,6] = x * (p[0]*x + p[2]*y + p[4] + x) / (p[6]*x + p[7]*y + p[8] + 1)**2
        dx[:,7] = y * (p[0]*x + p[2]*y + p[4] + x) / (p[6]*x + p[7]*y + p[8] + 1)**2
        dx[:,8] = 1.0 * (p[0]*x + p[2]*y + p[4] + x) / (p[6]*x + p[7]*y + p[8] + 1)**2

        dy[:,1] = x / (p[6]*x + p[7]*y + p[8] + 1) 
        dy[:,3] = y / (p[6]*x + p[7]*y + p[8] + 1) 
        dy[:,5] = 1.0 / (p[6]*x + p[7]*y + p[8] + 1) 
        dy[:,6] = x * (p[1]*x + p[3]*y + p[5] + y) / (p[6]*x + p[7]*y + p[8] + 1)**2
        dy[:,7] = y * (p[1]*x + p[3]*y + p[5] + y) / (p[6]*x + p[7]*y + p[8] + 1)**2
        dy[:,8] = 1.0 * (p[1]*x + p[3]*y + p[5] + y) / (p[6]*x + p[7]*y + p[8] + 1)**2

        return (dx, dy)

class ThinPlateSpline(Model):

    MODEL='Thin Plate Spline (TPS)'

    DESCRIPTION="""
        Computes a thin-plate-spline deformation model, as described in:
        
        Bookstein, F. L. (1989). Principal warps: thin-plate splines and the 
        decomposition of deformations. IEEE Transactions on Pattern Analysis 
        and Machine Intelligence, 11(6), 567-585. 
        
        """

    def __init__(self, coordinates):
    
        Model.__init__(self, coordinates)
    
    def U(self, r):
        """
        Kernel function, applied to solve the biharmonic equation.
        
        Parameters
        ----------
        r: float
            Distance between sample and coordinate point.
        
        Returns
        -------
        U: float
           Evaluated kernel.
        """
        
        return np.multiply(-np.power(r,2), np.log(np.power(r,2) + 1e-20))
    
    def fit(self, p0, p1, lmatrix=False):
        """
        Estimates the best fit parameters that define a warp field, which 
        deforms feature points p0 to p1.
        
        Parameters
        ----------
        p0: nd-array
            Image features (points).
        p1: nd-array
            Template features (points).
        lmatrix: boolean
            Enables the spline design matrix when returning.
            
        Returns
        -------
        parameters: nd-array
            Model parameters.
        error: float
            Sum of RMS error between p1 and alinged p0.
        L: nd-array
            Spline design matrix, optional (using lmatrix keyword).
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
        
        parameters = np.dot(np.linalg.inv(L), Y)
        
        # Estimate the thin-plate spline basis.
        self.__basis(p0)
        
        # Estimate the model fit error.
        _p0, _p1, _projP0, error = self.__splineError(p0, p1, parameters)
        
        if lmatrix:
            return parameters, error, L
        else:
            return parameters, error
    
    def __splineError(self, p0, p1, parameters):
        """
        Estimates the point alignment and computes the alignment error.
        
        Parameters
        ----------
        p0: nd-array
            Image features (points).
        p1: nd-array
            Template features (points).
        parameters: nd-array
            Thin-plate spline parameters.
        
        Returns
        -------
        error: float
            Alignment error between p1 and projected p0 (RMS).
        """
        
        # like __basis, compute a reduced set of basis vectors.
        
        basis = np.zeros((p0.shape[0], len(p0)+3))
        
        # nonlinear, spline component.
        for index, p in enumerate( p0 ):
            basis[:,index] = self.U( 
                np.sqrt(
                    (p[0]-p1[:,0])**2 + 
                    (p[1]-p1[:,1])**2 
                    )     
                ).flatten()

        # linear, affine component
        basis[:,-3] = 1
        basis[:,-2] = p1[:,1]
        basis[:,-1] = p1[:,0]
        
        # compute the alignment error.
        
        projP0 = np.vstack( [
           np.dot(basis, parameters[:,1]),
           np.dot(basis, parameters[:,0])
           ]
           ).T
        
        error = np.sqrt( 
           (projP0[:,0] - p1[:,0])**2 + (projP0[:,1] - p1[:,1])**2 
           ).sum() 
        
        return p0, p1, projP0, error
     
    def __basis(self, p0):
        """
        Forms the thin plate spline deformation basis, which is composed of 
        a linear and non-linear component.
        
        Parameters
        ----------
        p0: nd-array
            Image features (points).
        """
        
        self.basis = np.zeros((self.coordinates.tensor[0].size, len(p0)+3))
        
        # nonlinear, spline component.
        for index, p in enumerate( p0 ):
            self.basis[:,index] = self.U( 
                np.sqrt(
                    (p[0]-self.coordinates.tensor[1])**2 + 
                    (p[1]-self.coordinates.tensor[0])**2 
                    )     
            ).flatten()

        # linear, affine component
        
        self.basis[:,-3] = 1
        self.basis[:,-2] = self.coordinates.tensor[1].flatten()
        self.basis[:,-1] = self.coordinates.tensor[0].flatten()
    
    
    def transform(self, parameters):
        """
        A "thin-plate-spline" transformation of coordinates.
        
        Parameters
        ----------
        parameters: nd-array
            Model parameters.
        
        Returns
        -------
        coords: nd-array
           Deformation coordinates.
        """
        
        shape = self.coordinates.tensor[0].shape
        
        return np.array( [ np.dot(self.basis, parameters[:,1]).reshape(shape),
                           np.dot(self.basis, parameters[:,0]).reshape(shape)
                         ]
                       )
    
    def warp(self, parameters):
        """
        Computes the warp field given model parameters.
        
        Parameters
        ----------
        parameters: nd-array
            Model parameters.
        
        Returns
        -------
        warp: nd-array
           Deformation field.
        """
        
        return self.transform(parameters)
    
    
    def jacobian(self, p=None):
        raise NotImplementedError("""
            It does not make sense to use a non-linear optimization to 
            fit a thin-plate-spline model. Try the "CubicSpline" deformation
            model instead.
            """
            )
    
    @property
    def identity(self):
        raise NotImplementedError('')
    
class CubicSpline(Model):

    MODEL='CubicSpline (CS)'

    DESCRIPTION="""
        Applies a cubic-spline deformation model, as described in:

        Kybic, J. and Unser, M. (2003). Fast parametric elastic image
        registration. IEEE Transactions on Image Processing, 12(11), 1427-1442.
        """

    def __init__(self, coordinates):

        Model.__init__(self, coordinates)
        self.__basis()

    @property
    def identity(self):
        return np.zeros(self.basis.shape[1]*2)

    @property
    def numberOfParameters(self):
        return self.basis.shape[1]

    def __basis(self, order=4, divisions=4):
        """
        Computes the spline tensor product and stores the products, as basis
        vectors.
    
        Parameters
        ----------
        order: int
            B-spline order, optional.
        divisions: int, optional.
            Number of spline knots.
        """

        shape = self.coordinates.tensor[0].shape
        grid = self.coordinates.tensor

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
        """
        Estimates the best fit parameters that define a warp field.
        
        Parameters
        ----------
        warp: nd-array
            A deformation field.
            
        Returns
        -------
        parameters: nd-array
           Model parameters.
        """
        
        return np.hstack(
            (
             np.dot(np.linalg.pinv(self.basis),
                    (self.coordinates.tensor[0] - warp[0]).flatten()),
             np.dot(np.linalg.pinv(self.basis),
                    (self.coordinates.tensor[1] - warp[1]).flatten()),
            )
           ).T


    def warp(self, parameters):
        """
        Computes the warp field given model parameters.
        
        Parameters
        ----------
        parameters: nd-array
            Model parameters.
        
        Returns
        -------
        warp: nd-array
           Deformation field.
        """
        
        dwarp = self.transform(parameters)
        
        return self.coordinates.tensor - dwarp

    def transform(self, p):
        """
        Applies an spline transformation to image coordinates.
        
        Parameters
        ----------
        parameters: nd-array
            Model parameters.
        
        Returns
        -------
        coordinates: nd-array
           Deformation coordinates.
        """

        px = np.array(p[0:self.numberOfParameters])
        py = np.array(p[self.numberOfParameters::])

        shape = self.coordinates.tensor[0].shape

        return np.array( [ np.dot(self.basis, py).reshape(shape),
                           np.dot(self.basis, px).reshape(shape)
                         ]
                       )

    def jacobian(self, p=None):
        """
        Evaluate the derivative of deformation model with respect to the
        coordinates.
        """

        dx = np.zeros((self.coordinates.tensor[0].size,
                       2*self.numberOfParameters))

        dy = np.zeros((self.coordinates.tensor[0].size,
                       2*self.numberOfParameters))

        dx[:, 0:self.numberOfParameters] = self.basis
        dy[:, self.numberOfParameters::] = self.basis

        return (dx, dy)
