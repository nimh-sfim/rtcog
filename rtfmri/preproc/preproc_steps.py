import os.path as osp
import numpy as np

from utils.log import get_logger
from utils.exceptions import VolumeOverflowError
from utils.rt_functions import gen_polort_regressors
from utils.rt_functions import rt_EMA_vol, rt_regress_vol, rt_kalman_vol, rt_smooth_vol, rt_snorm_vol
from utils.fMRI import unmask_fMRI_img

log = get_logger()

class PreprocStep:
    """
    Base class for a preprocessing step in the real-time fMRI pipeline.

    Subclasses must implement the `_run(pipeline)` method, which is called on each TR 
    and receives access to the pipeline’s state. Optionally, subclasses may also implement 
    `_start(pipeline)` to initialize state before the first TR, and `_save(pipeline)` to 
    perform any custom saving logic at the end of the run.

    Attributes
    ----------
    save : bool
        Whether to save the output from this step to disk.
    suffix : str or None
        Filename suffix to use for saving a NIfTI file (if `save=True`).
    data_out : np.ndarray or None
        Cached 2D array of shape (N_voxels, N_timepoints) storing output for each TR, 
        populated if `save=True`.

    Class Attributes
    ----------------
    registry : dict
        Mapping of registered step names (e.g., "ema", "iglm") to class objects. Automatically
        populated via `__init_subclass__`.

    Methods
    -------
    start_step(pipeline):
        Optional setup logic to initialize internal state before processing begins.
    run(pipeline):
        Executes the step’s logic on the current TR. Saves result to `data_out` if saving is enabled.
    save_nifti(pipeline):
        Saves the accumulated data to disk as a NIfTI file using the provided mask and filename.
    get_class(name):
        Class method that retrieves a registered step class by name (case-insensitive).
    """
    registry = {} # Holds all available step classes that can be instantiated later on in Pipeline.

    def __init__(self, save=False, suffix=None):
        self.save = save
        self.suffix = suffix

        self.data_out = None
        
    @property
    def name(self):
        return self.__class__.__name__.replace('Step', '').lower()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__.replace('Step', '').lower()
        cls.registry[name] = cls
    
    def __eq__(self, other):
        """Allows lookup of object in pipeline.steps list"""
        if isinstance(other, str):
            return other.lower() in (self.__class__.__name__.lower(), 
                                     self.__class__.__name__.replace('Step', '').lower())
        elif isinstance(other, PreprocStep):
            return self.__class__ == other.__class__
        return False

    @classmethod
    def get_class(cls, name):
        return cls.registry.get(name.lower())

    def start_step(self, pipeline):
        if self.save:
            self.data_out = np.zeros((pipeline.Nv, pipeline.Nt))
        self._start(pipeline)

    def run(self, pipeline):
        output = self._run(pipeline)
        if self.save:
            self.data_out[:, pipeline.t] = output[:, 0]

        return output
    
    def save_nifti(self, pipeline):
        if self.save and self.suffix:
            out_path = osp.join(pipeline.out_dir, pipeline.out_prefix + self.suffix)
            unmask_fMRI_img(self.data_out, pipeline.mask_img, out_path)
            self._save(pipeline)

    def _start(self, pipeline):
        pass

    def _run(self, pipeline):
        raise NotImplementedError

    def _save(self, pipeline):
        pass
    
    def snapshot(self):
        """Return data_out for testing purposes"""
        return {self.name: self.data_out}
        
        
        




class EMAStep(PreprocStep):
    def __init__(self, save, suffix='.pp_EMA.nii', ema_th=0.98):
        super().__init__(save, suffix)
        self.EMA_th = ema_th
        self.EMA_filt = None

    def _run(self, pipeline):
        Data_FromAFNI = pipeline.Data_FromAFNI[:, :pipeline.t + 1]
        ema_data_out, self.EMA_filt = rt_EMA_vol(pipeline.n, self.EMA_th, Data_FromAFNI, self.EMA_filt)
        
        return ema_data_out
    

class iGLMStep(PreprocStep):
    def __init__(self, save, suffix='.pp_iGLM.nii'):
        super().__init__(save, suffix)
        self.iGLM_prev = {}

        if self.save:
            self.iGLM_Coeffs = None
        
    def _start(self, pipeline):
        if pipeline.iGLM_motion:
            self.iGLM_num_regressors = pipeline.iGLM_polort + 6
            self.nuisance_labels = ['Polort'+str(i) for i in np.arange(pipeline.iGLM_polort)] + ['roll','pitch','yaw','dS','dL','dP']
        else:
            self.iGLM_num_regressors = pipeline.iGLM_polort
            self.nuisance_labels = ['Polort'+str(i) for i in np.arange(pipeline.iGLM_polort)]
        
        if pipeline.iGLM_polort > -1:
            self.legendre_pols = gen_polort_regressors(pipeline.iGLM_polort, pipeline.Nt)
        else:
            self.legendre_pols = None

        if self.save:
            self.iGLM_Coeffs = np.zeros((pipeline.Nv, self.iGLM_num_regressors, pipeline.Nt))

    def _run(self, pipeline): 
        try:
            if pipeline.iGLM_motion:
                this_t_nuisance = np.concatenate((self.legendre_pols[pipeline.t,:], pipeline.motion))[:,np.newaxis]
            else:
                this_t_nuisance = (self.legendre_pols[pipeline.t,:])[:,np.newaxis]
        except IndexError:
            raise VolumeOverflowError()
            
        iGLM_data_out, self.iGLM_prev, Bn = rt_regress_vol(
            pipeline.n, 
            pipeline.processed_tr,
            this_t_nuisance,
            self.iGLM_prev,
        )

        if self.save:
            self.iGLM_Coeffs[:, :, pipeline.t] = np.squeeze(Bn, axis=2)

        return iGLM_data_out

    def _save(self, pipeline):
        if self.save:
            for i, lab in enumerate(self.nuisance_labels):
                data = self.iGLM_Coeffs[:,i,:]
                unmask_fMRI_img(data, pipeline.mask_img, osp.join(pipeline.out_dir, pipeline.out_prefix+'.pp_iGLM_'+lab+'.nii'))


class KalmanStep(PreprocStep):
    def __init__(self, save, suffix='.pp_LPfilter.nii'):
        super().__init__(save, suffix)
        self.S_x = None
        self.S_P = None
        self.fPositDerivSpike = None
        self.fNegatDerivSpike = None
        
    def _start(self, pipeline):
        self.S_x = np.zeros(pipeline.Nv)
        self.S_P = np.zeros(pipeline.Nv) 
        self.fPositDerivSpike = np.zeros(pipeline.Nv)
        self.fNegatDerivSpike = np.zeros(pipeline.Nv)

    def _run(self, pipeline):
        klm_data_out, S_x_new, S_P_new, fPos_new, fNeg_new = rt_kalman_vol(
            pipeline.n,
            pipeline.t,
            pipeline.processed_tr,
            pipeline.welford_std,
            self.S_x,
            self.S_P,
            self.fPositDerivSpike,
            self.fNegatDerivSpike,
            pipeline.n_cores,
            pipeline.pool,
        )

        self.S_x[:] = S_x_new.ravel()
        self.S_P[:] = S_P_new.ravel()
        self.fPositDerivSpike[:] = fPos_new.ravel()
        self.fNegatDerivSpike[:] = fNeg_new.ravel()

        if self.save:
            pipeline.Data_kalman[:, pipeline.t] = klm_data_out.squeeze()

        return klm_data_out


class SmoothStep(PreprocStep):
    def __init__(self, save=False, suffix='.pp_Smooth.nii'):
        super().__init__(save, suffix)

    def _run(self, pipeline):
        smooth_out = rt_smooth_vol(pipeline.processed_tr, pipeline.mask_img, fwhm=pipeline.FWHM)
            
        return smooth_out

    
class SnormStep(PreprocStep):
    def _run(self, pipeline):
        norm_out = rt_snorm_vol(pipeline.processed_tr)
        
        return norm_out
    

# Avoid use of string literals in pipeline.py
EMA = 'ema'
IGLM = 'iglm'
KALMAN = 'kalman'
SMOOTH = 'smooth'
SNORM = 'snorm'