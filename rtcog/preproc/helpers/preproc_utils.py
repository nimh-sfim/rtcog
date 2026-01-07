import numpy as np
from scipy.special import legendre
from scipy import ndimage
import logging
from scipy.signal.windows import exponential

from rtcog.utils.fMRI import unmask_fMRI_img, mask_fMRI_img

log = logging.getLogger('online_preproc')

def gen_polort_regressors(polort, nt):
    """
    Generate Legendre polynomials of a given order for nuisance regression purposes.

    Parameters
    ----------
    polort : int
        Maximum polynomial order to generate.
    nt : int
        Number of data points.

    Returns
    -------
    out : np.ndarray, shape (nt, polort)
        Legendre polynomial regressors.
    """ 
    out = np.zeros((nt, polort))
    for n in range(polort):
        Pn = legendre(n)
        x  = np.linspace(-1.0, 1.0, nt)
        out[:,n] = Pn(x).T
    return(out)


# Kalman Filter Functions
# =======================
def welford(k, x, M, S):
    # inputs np.array(Nv,)
    Mnext = M + (x-M)/k
    Snext = S + (x-M)*(x-Mnext)
    if k == 1:
        std = np.zeros(x.shape[0])
    else:
            std = np.sqrt(Snext/(k-1))
    return Mnext, Snext, std


# Smoothing Functions
# ===================
def _smooth_array(arr, affine, fwhm=None, ensure_finite=True, copy=True):
    """
    Smooth images by applying a Gaussian filter.
    Apply a Gaussian filter along the three first dimensions of arr.

    Based on https://github.com/nilearn/nilearn/blob/main/nilearn/image/image.py

    Parameters
    ----------
    arr : numpy.ndarray
        4D array, with image number as last dimension. 3D arrays are also
        accepted.
    affine : numpy.ndarray
        (4, 4) matrix, giving affine transformation for image. (3, 3) matrices
        are also accepted (only these coefficients are used).
        If fwhm='fast', the affine is not used and can be None
    fwhm : scalar, numpy.ndarray/tuple/list, 'fast' or None
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
    ensure_finite : bool
        if True, replace every non-finite values (like NaNs) by zero before
        filtering.
    copy : bool
        if True, input array is not modified. True by default: the filtering
        is not performed in-place.
    Returns
    -------
    filtered_arr : numpy.ndarray
        arr, filtered.
    Notes
    -----
    This function is most efficient with arr in C order.
    """
    # Here, we have to investigate use cases of fwhm. Particularly, if fwhm=0.
    # See issue #1537
    if isinstance(fwhm, (int, float)) and (fwhm == 0.0):
        log.warning("The parameter 'fwhm' for smoothing is specified "
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

def rt_smooth_vol(data_arr, mask_img, fwhm=4):
    """
    Apply smoothing to fMRI data volumes.

    Parameters
    ----------
    data_arr : np.ndarray, shape (Nvoxels, 1)
        The input fMRI data array to be processed.
    mask_img : nibabel.Nifti1Image
        A binary mask image.
    fwhm : float, optional
        The full width at half maximum (FWHM) value used to define the smoothing kernel. 
        The default is 4 mm.

    Returns
    -------
    np.ndarray, shape (Nvoxels, 1)
        The smoothed fMRI data.
    """
    data_arr = data_arr[:, 0]
    x      = unmask_fMRI_img(data_arr, mask_img)
    x_sm   = _smooth_array(x, affine=mask_img.affine, fwhm=fwhm)
    x_sm_v = mask_fMRI_img(x_sm, mask_img)

    return x_sm_v[:, np.newaxis]

# Temporal Normalization
# ==========================
def calculate_spc(current_signal, baseline_signal, mean_removed):
    """
    Calculate signal percent change (SPC) for a given timepoint.

    The SPC is computed voxel-wise as either:
    - `100 * (s / s̄)` if the mean has already been removed (e.g. via EMA)
    - `100 * ((s - s̄) / s̄)` otherwise

    Parameters
    ----------
    current_signal : np.ndarray, shape (Nvoxels,)
        The current volume's voxel intensities after prior preprocessing.
    baseline_signal : np.ndarray, shape (Nvoxels,)
        The voxel-wise mean signal computed from the baseline period.
    mean_removed : bool
        Whether the signal mean has already been removed (e.g., via EMA).
        If True, use multiplicative SPC; if False, use standard SPC.

    Returns
    -------
    np.ndarray, shape (Nvoxels, 1)
        The normalized signal.
    """
    if mean_removed:
        spc = (current_signal / baseline_signal) * 100
    else:
        spc = ((current_signal - baseline_signal) / baseline_signal) * 100
            
    return spc[:, np.newaxis]


# Windowing utilities
# ==========================
def create_win(M, center=0, tau=3):
    win = exponential(M, center, tau, False)
    return win[:, np.newaxis]

class CircularBuffer:
    def __init__(self, Nv, size):
        self.insert_idx = 0
        self.size = size
        self.buffer = np.zeros((Nv, size))
        self.full = False
    
    def update(self, data):
        self.buffer[:, self.insert_idx] = data.squeeze()
        self.insert_idx = (self.insert_idx + 1) % self.size

        if self.insert_idx == 0 and not self.full:
            self.full = True

        if self.full:
            return np.concatenate([self.buffer[:, self.insert_idx:], self.buffer[:, :self.insert_idx]], axis=1)
        
        else:
            return None
