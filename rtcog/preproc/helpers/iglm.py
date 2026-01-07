import numpy as np
from numpy.linalg import cholesky, inv

class iGLM:
    """
    Incremental Generalized Linear Model for detrending and removal of nuisance regressors in real time.
    """
    
    def __init__(self):
        self.n = 1
        self.Cn = None  # Matrix for Cholesky decomposition
        self.Dn = None  # Dn matrix
        self.s2n = None  # Sigma square estimates
    
    def _iGLMVol(self, n, Yn, Fn, Dn, Cn, s2n):
        """
        Internal method for incremental GLM computation.
        
        Implementation of algorithm described in:
        Bagarinao, E., Matsuo, K., Nakai, T., Sato, S., 2003. Estimation of
        general linear model coefficients for real-time application. NeuroImage
        19, 422-429.

        Parameters
        ----------
        n : int
            Current volume number.
        Yn : np.ndarray, shape (Nvoxels, 1)
            Current masked data point.
        Fn : np.ndarray, shape (Nregressors, 1)
            Current regressor values.
        Dn : np.ndarray
            Sum of (Yn * Ft') at time n-1.
        Cn : np.ndarray
            Matrix for Cholesky decomposition at time n-1.
        s2n : np.ndarray
            Sigma square estimate for n-1.
        
        Returns
        -------
        Bn : np.ndarray
            Current estimates for linear regression coefficients.
        Cn : np.ndarray
            Updated matrix for Cholesky decomposition.
        Dn : np.ndarray
            Updated Dn matrix.
        s2n : np.ndarray
            Updated sigma square estimate.
        """
        nv = Yn.shape[0]
        nrBasFct = Fn.shape[0]

        Dn = Dn + np.matmul(Yn, Fn.T)  # Eq. 17
        Cn = (((n-1)/n) * Cn) + ((1/n)*np.matmul(Fn, Fn.T))  # Eq. 18
        s2n = s2n + (Yn * Yn)  # Eq. 9 without the 1/n factor
        
        is_pos_def = np.all(np.linalg.eigvals(Cn) > 1e-10)
        if (is_pos_def) and (n > nrBasFct + 2):
            Nn = cholesky(Cn).T
            iNn = inv(Nn.T)
            An = (1/n) * np.matmul(Dn, iNn.T)  # Eq. 14
            Bn = np.matmul(An, iNn)  # Eq. 16
        else:
            Bn = np.zeros((nv, nrBasFct))
        return Bn, Cn, Dn, s2n
    
    def regress_vol(self, n, Yn, Fn):
        """
        Apply real-time regression to fMRI data.
        
        Parameters
        ----------
        n : int
            Current volume number.
        Yn : np.ndarray, shape (Nvoxels, 1)
            The current fMRI data point (masked).
        Fn : np.ndarray, shape (Nregressors, 1)
            The current regressor values.
        
        Returns
        -------
        Yn_d : np.ndarray, shape (Nvoxels, 1)
            The residual (detrended) fMRI data.
        Bn : np.ndarray, shape (Nvoxels, Nregressors, 1)
            The regression coefficient estimates.
        """
        if n == 1:
            L = Fn.shape[0]  # Number of Regressors
            Nv = Yn.shape[0]
            self.Cn = np.zeros((L, L), dtype='float64')  # Matrix for Cholesky decomposition
            self.Dn = np.zeros((Nv, L), dtype='float64')  # Dn
            self.s2n = np.zeros((Nv, 1), dtype='float64')  # Sigma estimates
        
        Bn, self.Cn, self.Dn, self.s2n = self._iGLMVol(n, Yn, Fn, self.Dn, self.Cn, self.s2n)
        Yn_d = Yn - np.matmul(Bn, Fn)
        Yn_d = np.squeeze(Yn_d)
        
        return Yn_d[:, np.newaxis], Bn[:, :, np.newaxis]
