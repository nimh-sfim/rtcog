import os.path as osp
import numpy as np

from utils.log import get_logger
from core.exceptions import VolumeOverflowError
from utils.rt_functions import gen_polort_regressors
from utils.rt_functions import rt_EMA_vol, rt_regress_vol, rt_kalman_vol, rt_smooth_vol, rt_snorm_vol
from utils.fMRI import unmask_fMRI_img

log = get_logger()

class PreprocStep:
    """
    Base class for preprocessing steps in the real-time fMRI pipeline.

    Each subclass must implement a `run(pipeline)` method, which:
      - Reads `pipeline.processed_tr` (a numpy array of shape (N_voxels, 1))
      - Performs its step-specific transformation
      - Returns a numpy array of the same shape
      
    Each subclass can also implement `start_step(pipeline)` to initlialize any variables 
    during the first TR that require access to pipeline's attributes.
    
    You can also implement `run_discard(pipeline)` and `save_nifti(pipeline)` if you want to
    save its nifti file at the end of the run.

    Limitations:
      - Input/output data shape must be (N_voxels, 1).
      - Will raise runtime errors if this condition is not met.
    """
    registry = {} # Holds all available step classes that can be instantiated later on in Pipeline.

    def __init__(self, save=False):
        self.save = save

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
        """Optional: Define variables needed before processing begins (e.g., for saving data)."""
        # NOTE: can use setattr to define variable names dynamically to make this more easily subclassed, but might be too confusing?
        pass

    def run_discard_volumes(self, pipeline):
        """Optional: Handle discarded volumes (e.g., append zeros to arrays)."""
        pass
    
    def run(self, pipeline):
        raise NotImplementedError
    
    def save_nifti(self, pipeline):
        if self.save:
            raise NotImplementedError

    @staticmethod
    def prep_file(data, file_suffix, pipeline):
        out_path = osp.join(pipeline.out_dir, pipeline.out_prefix+file_suffix)
        unmask_fMRI_img(data, pipeline.mask_img, out_path)


class EMAStep(PreprocStep):
    def __init__(self, save, ema_th=0.98):
        super().__init__(save)
        self.EMA_th = ema_th
        self.EMA_filt = None

    def start_step(self, pipeline):
        if self.save:
            pipeline.Data_EMA = np.zeros((pipeline.Nv, pipeline.Nt))
    
    def run_discard_volumes(self, pipeline):
        if self.save:
            pipeline.Data_EMA[:, pipeline.t] = np.zeros((pipeline.Nv,))

    def run(self, pipeline):
        ema_data_out, self.EMA_filt = rt_EMA_vol(pipeline.n, self.EMA_th, pipeline.Data_FromAFNI, self.EMA_filt)
        if self.save: 
            pipeline.Data_EMA[:, pipeline.t] = ema_data_out.squeeze()
        
        return ema_data_out
    
    def save_nifti(self, pipeline):
        if self.save:
            self.prep_file(pipeline.Data_EMA, '.pp_EMA.nii', pipeline)
    

class iGLMStep(PreprocStep):
    def __init__(self, save):
        super().__init__(save)
        self.iGLM_prev = {}

        if self.save:
            self.iGLM_Coeffs = None
        
    def start_step(self, pipeline):
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
            pipeline.Data_iGLM = np.zeros((pipeline.Nv, pipeline.Nt))
            self.iGLM_Coeffs = np.zeros((pipeline.Nv, self.iGLM_num_regressors, pipeline.Nt))

    def run_discard_volumes(self, pipeline):
        if self.save:
            pipeline.Data_iGLM[:, pipeline.t] = np.zeros((pipeline.Nv,))
            self.iGLM_Coeffs[:, :, pipeline.t] = 0
            
    def run(self, pipeline): 
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
            pipeline.Data_iGLM[:, pipeline.t] = iGLM_data_out.squeeze()
            self.iGLM_Coeffs[:, :, pipeline.t] = np.squeeze(Bn, axis=2)

        return iGLM_data_out

    def save_nifti(self, pipeline):
        if self.save:
            self.prep_file(pipeline.Data_iGLM, '.pp_iGLM.nii', pipeline)
            for i, lab in enumerate(self.nuisance_labels):
                data = self.iGLM_Coeffs[:,i,:]
                unmask_fMRI_img(data, pipeline.mask_img, osp.join(pipeline.out_dir, pipeline.out_prefix+'.pp_iGLM_'+lab+'.nii'))


class KalmanStep(PreprocStep):
    def __init__(self, save):
        super().__init__(save)
        self.S_x = None
        self.S_P = None
        self.fPositDerivSpike = None
        self.fNegatDerivSpike = None
        
    def start_step(self, pipeline):
        self.S_x = np.zeros(pipeline.Nv)
        self.S_P = np.zeros(pipeline.Nv) 
        self.fPositDerivSpike = np.zeros(pipeline.Nv)
        self.fNegatDerivSpike = np.zeros(pipeline.Nv)

        if self.save:
            pipeline.Data_kalman = np.zeros((pipeline.Nv, pipeline.Nt))
    
    def run_discard_volumes(self, pipeline):
        if self.save:
            pipeline.Data_kalman[:, pipeline.t] = np.zeros((pipeline.Nv,))

    def run(self, pipeline):
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

    def save_nifti(self, pipeline):
        if self.save:
            self.prep_file(pipeline.Data_kalman, '.pp_LPfilter.nii', pipeline)


class SmoothStep(PreprocStep):
    def start_step(self, pipeline):
        if self.save:
            pipeline.Data_smooth = np.zeros((pipeline.Nv, pipeline.Nt))

    def run_discard_volumes(self, pipeline):
        if self.save:
            pipeline.Data_smooth[:, pipeline.t] = np.zeros((pipeline.Nv,))

    def run(self, pipeline):
        smooth_out = rt_smooth_vol(pipeline.processed_tr, pipeline.mask_img, fwhm=pipeline.FWHM)
        if self.save:
            pipeline.Data_smooth[:, pipeline.t] = smooth_out.squeeze()
            
        return smooth_out
    
    def save_nifti(self, pipeline):
        if self.save:
            self.prep_file(pipeline.Data_smooth, '.pp_Smooth.nii', pipeline)


class SnormStep(PreprocStep):
    def start_step(self, pipeline):
        if self.save:
            pipeline.Data_norm = np.zeros((pipeline.Nv, pipeline.Nt))

    def run_discard_volumes(self, pipeline):
        if self.save:
            pipeline.Data_norm[:, pipeline.t] = np.zeros((pipeline.Nv,))

    def run(self, pipeline):
        norm_out = rt_snorm_vol(pipeline.processed_tr)
        if self.save:
            pipeline.Data_norm[:, pipeline.t] = norm_out.squeeze()
        
        return norm_out
    
    def save_nifti(self, pipeline):
        if self.save:
            self.prep_file(pipeline.Data_norm, '.pp_Snorm.nii', pipeline)
    

# Avoid use of string literals in pipeline.py
EMA = 'ema'
IGLM = 'iglm'
KALMAN = 'kalman'
SMOOTH = 'smooth'
SNORM = 'snorm'