from .fMRI import unmask_fMRI_img, mask_fMRI_img
import numpy as np
from scipy.special import legendre
from scipy import ndimage
import itertools
from numpy.linalg import cholesky, inv
import logging as log
from sklearn.preprocessing import StandardScaler

log.basicConfig(format='[%(levelname)s]: %(message)s', level=log.DEBUG)
def init_iGLM():
    return 1, {}

def gen_polort_regressors(polort, nt):
    """Generate Legendre Polynomials of a given order for nuisance regression purposes
    
    Parameters:
    -----------
    polort: int
        maximum polynomial order to generate
    nt: int
        number of data points
        
    Returns:
    --------
    out: np.array [Nt,Nregressors]
        legendre polynomials
    """
    
    out = np.zeros((nt, polort))
    for n in range(polort):
        Pn = legendre(n)
        x  = np.linspace(-1.0, 1.0,nt)
        out[:,n] = Pn(x).T
    return(out)

def _is_pos_def(x):
    """ Checks if a matrix is positive definitive. This is needed for the cholesky decomposition
    
    Parameters:
    -----------
    x: np.array
        Square matrix
        
    Returns:
    --------
    out: boolean
    """
    return np.all(np.linalg.eigvals(x) > 1e-10)

def _iGLMVol(n,Yn,Fn,Dn,Cn,s2n):
    """ Incremental GLM for detrending and removal of nuisance regressors in realtime.
    Implementation of algorithm described in:
    Bagarinao, E., Matsuo, K., Nakai, T., Sato, S., 2003. Estimation of
    general linear model coefficients for real-time application. NeuroImage
    19, 422-429.
    
    Parameters:
    -----------
    n: int
        Current volume number entering the GLM. Not to be confused with the current
        acquisition number
    Yn: np.array [Nvoxels,1]
        Current masked data point
    Fn: np.array [Nregressors,1]
        Current regressor values (e.g., motion, trends, etc.)
    Dn: np.array []
        sum of (Yn * Ft') at time n-1  (Eq. 17) 
    Cn: np.array []
        matrix for Cholesky decomposition at time n-1 (Eq. 15)
    s2n: np.array []
        sigma square estimate for n-1 
    Returns:
    --------
    
    Bn: np.array []
        current estimates for linear regression coefficients at time n
    Cn: np.array []
        updated estimates of matrix for Cholesky decomposition at time n
    Dn: np.array []
        updated estimate of Dn matrix at time n
    s2n: np.array []
        updated estimate of sigma square at time n
    """    
    
    nv       = Yn.shape[0]
    nrBasFct = Fn.shape[0]  
    df = n - nrBasFct                        # Degrees of Freedom
    Dn = Dn + np.matmul(Yn,Fn.T)             # Eq. 17
    Cn = (((n-1)/n) * Cn) + ((1/n)*np.matmul(Fn,Fn.T))  # Eq. 18
    s2n = s2n + (Yn * Yn)            # Eq. 9 without the 1/n factor... see below
    if (_is_pos_def(Cn) == True) and (n > nrBasFct + 2):
        Nn = cholesky(Cn).T
        iNn = inv(Nn.T)
        An    = (1/n) * np.matmul(Dn,iNn.T)  # Eq. 14
        Bn    = np.matmul(An,iNn)            # Eq. 16
    else:
        #print ('%d non positive definite'% n)
        Bn = np.zeros((nv,nrBasFct))
    return Bn,Cn,Dn,s2n

def rt_regress_vol(n,Yn,Fn,prev, do_operation = True):
    if not do_operation:
        L   = Fn.shape[0]                               # Number of Regressors
        Nv  = Yn.shape[0]
        b_out = np.zeros((Nv,L,1))
        p_out = {'Cn':None, 'Dn':None,'s2n':None}
        y_out = Yn
        return y_out, p_out, b_out
    if n == 1:
        L   = Fn.shape[0]                               # Number of Regressors
        Nv  = Yn.shape[0]
        Cn  = np.zeros((L,L), dtype='float64')                  # Matrix for Cholesky decomposition
        Dn  = np.zeros((Nv, L), dtype='float64')   # Dn
        s2n = np.zeros((Nv, 1), dtype='float64')   # Sigma Estimates
    else:
        Cn  = prev['Cn']
        Dn  = prev['Dn']
        s2n = prev['s2n']
    
    Bn,Cn,Dn,s2n = _iGLMVol(n,Yn,Fn,Dn,Cn,s2n)
    Yn_d         = Yn - np.matmul(Bn,Fn)
    Yn_d         = np.squeeze(Yn_d)
    
    new = {'Cn':Cn, 'Dn':Dn, 's2n': s2n}
    return Yn_d[:,np.newaxis], new, Bn[:,:,np.newaxis]


# EMA Related Functions
# =====================
def init_EMA():
    EMA_th      = 0.98
    EMA_filt    = None
    return EMA_th, EMA_filt

def _apply_EMA_filter(a,emaIn,filtInput):
    A            = (np.array([a,1-a])[:,np.newaxis]).T
    EMA_filt_out = np.dot(A, np.hstack([filtInput,emaIn]).T).T
    EMA_out      = emaIn - EMA_filt_out
    return EMA_out,EMA_filt_out

def rt_EMA_vol(n,t,th,data,filt_in, do_operation=True):
    if do_operation: # Apply this operation
        if n == 1:   # First step
            filt_out = data[:,t][:,np.newaxis]
            data_out = (data[:,t] - data[:,t-1])[:,np.newaxis] 
        else:        # Any other step
            data_out, filt_out = _apply_EMA_filter(th,data[:,t][:,np.newaxis],filt_in)
            #data_out = np.squeeze(data_out)
    else: # Do not apply this operation
        data_out = data[:,t][:,np.newaxis]
        filt_out = None
    return data_out, filt_out

# Kalman Filter Functions
# =======================
def init_kalman(Nv,Nt):
    S_x              = np.zeros((Nv, Nt))
    S_P              = np.zeros((Nv, Nt))
    fPositDerivSpike = np.zeros((Nv, Nt))
    fNegatDerivSpike = np.zeros((Nv, Nt))
    kalmThreshold    = np.zeros((Nv, Nt))
    return S_x, S_P, fPositDerivSpike, fNegatDerivSpike, kalmThreshold

def _kalman_filter(kalmTh, kalmIn, S, fPositDerivSpike, fNegatDerivSpike):
    """ Perform Kalman Low-pass filtering and despiking
    Based on: Koush Y., Zvyagintsev M., Dyck M., Mathiak K.A., Mathiak K. (2012)
    Signal quality and Bayesian signal processing in neurofeedback based on 
    real-time fMRI. Neuroimage 59:478-89
    
    Parameters:
    -----------
    kalmTh: 
        Spike-detection Threshold
    kalmIn:
        Input data
    S:
        Parameter structure
    fPositDerivSpike:
        Counter for spikes with positive derivative
    fNegatDerivSpike:
        Counter for spikes with negative derivative
    
    Returns:
    --------
    kalmOut:
        Filtered Output
    S:
        Parameter Structure
    fPositDerivSpike:
        Counter for spikes with positive derivative
    fNegatDerivSpike:
        Counter for spikes with negative derivative
    """
    
    # Preset
    A = 1
    H = 1
    I = 1
    #kalmOut = 0
    
    # Kalman Filter
    S['x'] = A * S['x']
    S['P'] = ( A * S['P'] * A ) + S['Q']
    if ((H*S['P']*H) + S['R']) > 0.0:
        K = S['P'] * H * (1/( (H*S['P']*H) + S['R']) )
    else:
        K = 0
    tmp_x  = S['x']
    tmp_p  = S['P']
    diff   = K * (kalmIn - (H*S['x']))
    S['x'] = S['x'] + diff
    S['P'] = (I - (K * H)) * S['P']
    # Spikes identification and correction
    if np.abs(diff) < kalmTh:
        #print('np.abs(diff) < kalmTh')
        kalmOut = H * S['x']
        fNegatDerivSpike = 0;
        fPositDerivSpike = 0;
    else:
        #print('np.abs(diff) > kalmTh')
        if diff > 0:
            if fPositDerivSpike < 1:
                kalmOut = H * tmp_x
                S['x'] = tmp_x
                S['P'] = tmp_p
                fPositDerivSpike = fPositDerivSpike + 1
            else:
                kalmOut = H * S['x']
                fPositDerivSpike = 0
        else:
            if fNegatDerivSpike < 1:
                kalmOut = H * tmp_x
                S['x'] = tmp_x
                S['P'] = tmp_p
                fNegatDerivSpike = fNegatDerivSpike + 1
            else:
                kalmOut = H * S['x']
                fNegatDerivSpike = 0

    return kalmOut,S,fPositDerivSpike,fNegatDerivSpike

def _kalman_filter_mv(input_dict):
    Nv = input_dict['vox'].shape[0]
    out_d_mv    = []
    out_fPos_mv = []
    out_fNeg_mv = []
    out_S_x_mv  = []
    out_S_P_mv  = []
    out_vox_mv  = []
    for v in np.arange(Nv):
        input_d = input_dict['d'][v]
        input_ts_STD = input_dict['std'][v]
        input_S = {'Q': input_dict['S_Q'][v],
                   'R': input_dict['S_R'][v],
                   'x': input_dict['S_x'][v],
                   'P': input_dict['S_P'][v]}
        input_fPos = input_dict['fPos'][v]
        input_fNeg = input_dict['fNeg'][v]
        kalmTh     = 0.9 * input_ts_STD
        [out_d, out_S, out_fPos, out_fNeg] = _kalman_filter(kalmTh, input_d,input_S, input_fPos, input_fNeg)
        for (l,i) in zip([out_d_mv,out_fPos_mv,out_fNeg_mv,out_S_x_mv,out_S_P_mv, out_vox_mv],
                         [out_d,   out_fPos,   out_fNeg,   out_S['x'],out_S['P'],input_dict['vox'][v]]):
            l.append(i)
    return [out_d_mv, out_fPos_mv, out_fNeg_mv, out_S_x_mv, out_S_P_mv, out_vox_mv]

def rt_kalman_vol(n,t,data,S_x,S_P,fPositDerivSpike,fNegatDerivSpike,num_cores,DONT_USE_VOLS,pool,do_operation=True):
    if n > 2 and do_operation==True:
        #log.debug('[t=%d,n=%d] rt_kalman_vol - Time to do some math' % (t, n))
        #log.debug('[t=%d,n=%d] rt_kalman_vol - Num Cores = %d' % (t, n, num_cores))
        #log.debug('[t=%d,n=%d] rt_kalman_vol - Input Data Dimensions %s' % (t, n, str(data.shape)))
        v_start = np.arange(0,data.shape[0],int(np.floor(data.shape[0]/(num_cores-1)))).tolist()
        v_end   = v_start[1:] + [data.shape[0]]
        #log.debug('[t=%d,n=%d] rt_kalman_vol - v_start %s' % (t, n, str(v_start)))
        #log.debug('[t=%d,n=%d] rt_kalman_vol - v_end   %s' % (t, n, str(v_end)))
        o_data, o_fPos, o_fNeg = [],[],[]
        o_S_x, o_S_P           = [],[]
        data_std               = np.std(data[:,DONT_USE_VOLS:t+1], axis=1)
        data_std_sq            = np.power(data_std,2) 
        inputs = ({'d'   : data[v_s:v_e,t],
                   'std' : data_std[v_s:v_e],
                   'S_x' : S_x[v_s:v_e],
                   'S_P' : S_P[v_s:v_e],
                   'S_Q' : 0.25 * data_std_sq[v_s:v_e],
                   'S_R' : data_std_sq[v_s:v_e],
                   'fPos': fPositDerivSpike[v_s:v_e],
                   'fNeg': fNegatDerivSpike[v_s:v_e],
                   'vox' : np.arange(v_s,v_e)}
                  for v_s,v_e in zip(v_start,v_end))
        #log.debug('[t=%d,n=%d] rt_kalman_vol - About to go parallel with %d cores' % (t, n, num_cores))
        res = pool.map(_kalman_filter_mv,inputs)
        #log.debug('[t=%d,n=%d] rt_kalman_vol - All parallel operationsc completed.' % (t, n))

        for j in np.arange(num_cores):
            o_data.append(res[j][0])
            o_fPos.append(res[j][1])
            o_fNeg.append(res[j][2])
            o_S_x.append(res[j][3])
            o_S_P.append(res[j][4])
        
        o_data = np.array(list(itertools.chain(*o_data)))[:,np.newaxis]
        o_fPos = np.array(list(itertools.chain(*o_fPos)))[:,np.newaxis]
        o_fNeg = np.array(list(itertools.chain(*o_fNeg)))[:,np.newaxis]
        o_S_x  = np.array(list(itertools.chain(*o_S_x)))[:,np.newaxis]
        o_S_P  = np.array(list(itertools.chain(*o_S_P)))[:,np.newaxis]
        
        #log.debug('[t=%d,n=%d] rt_kalman_vol - o_data.shape = %s' % (t, n, str(o_data.shape)))
        #o_data   = [item for sublist in o_data   for item in sublist]
        #o_fPos   = [item for sublist in o_fPos for item in sublist]
        #o_fNeg   = [item for sublist in o_fNeg for item in sublist]
        #o_S_x    = [item for sublist in o_S_x for item in sublist]
        #o_S_P    = [item for sublist in o_S_P for item in sublist]
        
        out     = [ o_data, o_S_x, o_S_P, o_fPos, o_fNeg ]
    else:
        #log.debug('[t=%d,n=%d] rt_kalman_vol - Skip this pass. Return empty containsers.' % (t, n))
        [Nv,Nt] = data.shape
        out     = [np.zeros((Nv,1)) for i in np.arange(5)]
    return out

# Smoothing Functions
# ===================
def _smooth_array(arr, affine, fwhm=None, ensure_finite=True, copy=True):
    """Smooth images by applying a Gaussian filter.
    Apply a Gaussian filter along the three first dimensions of arr.
    Parameters
    ----------
    arr: numpy.ndarray
        4D array, with image number as last dimension. 3D arrays are also
        accepted.
    affine: numpy.ndarray
        (4, 4) matrix, giving affine transformation for image. (3, 3) matrices
        are also accepted (only these coefficients are used).
        If fwhm='fast', the affine is not used and can be None
    fwhm: scalar, numpy.ndarray/tuple/list, 'fast' or None
        Smoothing strength, as a full-width at half maximum, in millimeters.
        If a nonzero scalar is given, width is identical in all 3 directions.
        A numpy.ndarray/list/tuple must have 3 elements,
        giving the FWHM along each axis.
        If any of the elements is zero or None,
        smoothing is not performed along that axis.
        If fwhm == 'fast', a fast smoothing will be performed with
        a filter [0.2, 1, 0.2] in each direction and a normalisation
        to preserve the local average value.
        If fwhm is None, no filtering is performed (useful when just removal
        of non-finite values is needed).
    ensure_finite: bool
        if True, replace every non-finite values (like NaNs) by zero before
        filtering.
    copy: bool
        if True, input array is not modified. True by default: the filtering
        is not performed in-place.
    Returns
    -------
    filtered_arr: numpy.ndarray
        arr, filtered.
    Notes
    -----
    This function is most efficient with arr in C order.
    """
    # Here, we have to investigate use cases of fwhm. Particularly, if fwhm=0.
    # See issue #1537
    if isinstance(fwhm, (int, float)) and (fwhm == 0.0):
        warnings.warn("The parameter 'fwhm' for smoothing is specified "
                      "as {0}. Setting it to None "
                      "(no smoothing will be performed)"
                      .format(fwhm))
        fwhm = None
    if arr.dtype.kind == 'i':
        if arr.dtype == np.int64:
            arr = arr.astype(np.float64)
        else:
            arr = arr.astype(np.float32)  # We don't need crazy precision.
    if copy:
        arr = arr.copy()
    if ensure_finite:
        # SPM tends to put NaNs in the data outside the brain
        arr[np.logical_not(np.isfinite(arr))] = 0
    if isinstance(fwhm, str) and (fwhm == 'fast'):
        arr = _fast_smooth_array(arr)
    elif fwhm is not None:
        fwhm = np.asarray(fwhm)
        fwhm = np.where(fwhm == None, 0.0, fwhm)  # noqa: E711
        affine = affine[:3, :3]  # Keep only the scale part.
        fwhm_over_sigma_ratio = np.sqrt(8 * np.log(2))  # FWHM to sigma.
        vox_size = np.sqrt(np.sum(affine ** 2, axis=0))
        sigma = fwhm / (fwhm_over_sigma_ratio * vox_size)
        for n, s in enumerate(sigma):
            if s > 0.0:
                ndimage.gaussian_filter1d(arr, s, output=arr, axis=n)
    return arr

def rt_smooth_vol(data_arr,mask_img,fwhm=4,do_operation=True):
    if do_operation:
        x      = unmask_fMRI_img(data_arr, mask_img)
        x_sm   = _smooth_array(x, affine=mask_img.affine, fwhm=fwhm)
        x_sm_v = mask_fMRI_img(x_sm, mask_img)
    else:
        x_sm_v = data_arr
    return x_sm_v[:, np.newaxis]

# Spatial Z-scoring Function
# ==========================
def rt_snorm_vol(data, do_operation=True):
    data = data[:,np.newaxis]
    sc  = StandardScaler(with_mean=True, with_std=True)
    out = sc.fit_transform(data)
    return out
