import numpy as np

from utils.rt_functions import rt_EMA_vol, rt_regress_vol, rt_kalman_vol, rt_smooth_vol, rt_snorm_vol
from utils.log import get_logger

log = get_logger()

class PreprocStep:
    """
    Base class for preprocessing steps in the real-time fMRI pipeline.

    Each subclass must implement a `run(pipeline)` method, which:
      - Reads `pipeline.processed_tr` (a numpy array of shape (N_voxels, 1))
      - Performs its step-specific transformation
      - Updates `pipeline.processed_tr` with a new numpy array of the same shape
      
    Each subclass can also implement `initialize_array(pipeline)` and `run_discard` if you want to save its nifti file
    at the end of the run.

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

    @classmethod
    def get_class(cls, name):
        return cls.registry.get(name.lower())

    def initialize_array(self, pipeline):
        # NOTE: can use setattr to define variable names dynamically to make this more easily subclassed, but might be too confusing?
        pass
    
    def run(self, pipeline):
        raise NotImplementedError

class EMAStep(PreprocStep):
    # def __init__(self, save, ema_th=0.98):
    #     super().__init__(save)
    #     self.ema_th = ema_th
    #     self.ema_filt = None

    def initialize_array(self, pipeline):
        if self.save:
            pipeline.Data_EMA = np.zeros((pipeline.Nv, 1))
            log.debug(f'[t={pipeline.t},n={pipeline.n}] Init - Data_EMA.shape      {pipeline.Data_EMA.shape}') 
    
    def run(self, pipeline):
        ema_data_out, pipeline.EMA_filt = rt_EMA_vol(pipeline.n, pipeline.EMA_th, pipeline.Data_FromAFNI, pipeline.EMA_filt)
        if pipeline.save_ema: 
            pipeline.Data_EMA = np.append(pipeline.Data_EMA, ema_data_out, axis=1)
            log.debug(f'[t={pipeline.t},n={pipeline.n}] Online - EMA - Data_EMA.shape      {pipeline.Data_EMA.shape}')
        
        pipeline.processed_tr = ema_data_out
    
class iGLMStep(PreprocStep):
    # def __init__(self, save):
    #     super().__init__(save)

    def initialize_array(self, pipeline):
        if self.save:
            pipeline.Data_iGLM = np.zeros((pipeline.Nv, 1))
            pipeline.iGLM_Coeffs = np.zeros((pipeline.Nv, pipeline.iGLM_num_regressors, 1))
            log.debug(f'[t={pipeline.t},n={pipeline.n}] Init - Data_iGLM.shape      {pipeline.Data_iGLM.shape}') 

    def run(self, pipeline): 
        if pipeline.iGLM_motion:
            this_t_nuisance = np.concatenate((pipeline.legendre_pols[pipeline.t,:], pipeline.motion))[:,np.newaxis]
        else:
            this_t_nuisance = (pipeline.legendre_pols[pipeline.t,:])[:,np.newaxis]
            
        iGLM_data_out, pipeline.iGLM_prev, Bn = rt_regress_vol(
            pipeline.n, 
            pipeline.processed_tr,
            this_t_nuisance,
            pipeline.iGLM_prev,
        )

        if pipeline.save_iGLM: 
            pipeline.Data_iGLM = np.append(pipeline.Data_iGLM, iGLM_data_out, axis=1)
            pipeline.iGLM_Coeffs = np.append(pipeline.iGLM_Coeffs, Bn, axis=2) 
            log.debug(f'[t={pipeline.t},n={pipeline.n}] Online - iGLM - Data_iGLM.shape     {pipeline.Data_iGLM.shape}')
            log.debug(f'[t={pipeline.t},n={pipeline.n}] Online - iGLM - iGLM_Coeffs.shape   {pipeline.iGLM_Coeffs.shape}')

        pipeline.processed_tr = iGLM_data_out

class KalmanStep(PreprocStep):
    # def __init__(self, save):
    #     super().__init__(save)
    #     self.S_x = None
    #     self.S_P = None
    #     self.fPositDerivSpike = None
    #     self.fNegatDerivSpike = None
    #     self.kalmThreshold = None

    def initialize_array(self, pipeline):
        if self.save:
            pipeline.Data_kalman = np.zeros((pipeline.Nv, 1))
            log.debug(f'[t={pipeline.t},n={pipeline.n}] Init - Data_kalman.shape   {pipeline.Data_kalman.shape}')

    def run(self, pipeline):
        klm_data_out, pipeline.S_x, pipeline.S_P, pipeline.fPositDerivSpike, pipeline.fNegatDerivSpike = rt_kalman_vol(
            pipeline.n,
            pipeline.t,
            pipeline.processed_tr,
            pipeline.welford_std,
            pipeline.S_x,
            pipeline.S_P,
            pipeline.fPositDerivSpike,
            pipeline.fNegatDerivSpike,
            pipeline.n_cores,
            pipeline.pool,
        )
        
        if pipeline.save_kalman: 
            pipeline.Data_kalman = np.append(pipeline.Data_kalman, klm_data_out, axis=1)
            log.debug(f'[t={pipeline.t},n={pipeline.n}] Online - Kalman - Data_kalman.shape     {pipeline.Data_kalman.shape}')
        
        pipeline.processed_tr = klm_data_out
    

class SmoothStep(PreprocStep):
    def initialize_array(self, pipeline):
        if self.save:
            pipeline.Data_smooth = np.zeros((pipeline.Nv, 1))

    def run(self, pipeline):
        smooth_out = rt_smooth_vol(pipeline.processed_tr, pipeline.mask_img, fwhm=pipeline.FWHM)
        if pipeline.save_smooth:
            pipeline.Data_smooth = np.append(pipeline.Data_smooth, smooth_out, axis=1)
            log.debug(f'[t={pipeline.t},n={pipeline.n}] Online - Smooth - Data_smooth.shape   {pipeline.Data_smooth.shape}')
            log.debug(f'[t={pipeline.t},n={pipeline.n}] Online - EMA - smooth_out.shape      {smooth_out.shape}')
            
        pipeline.processed_tr = smooth_out

class SnormStep(PreprocStep):
    def initialize_array(self, pipeline):
        # pipeline doesn't have save_snorm yet
        if self.save:
            pipeline.Data_norm = np.zeros((pipeline.Nv, 1))
            log.debug(f'[t={pipeline.t},n={pipeline.n}] Init - Data_norm.shape     {pipeline.Data_norm.shape}')

    def run(self, pipeline):
        # Should i make a pipeline.save_norm? I'm pretty sure this doesn't exist becuase it's the last step, so it was just
        # whatever was saved in the end...
        norm_out = rt_snorm_vol(pipeline.processed_tr)
    
        pipeline.Data_norm = np.append(pipeline.Data_norm, norm_out, axis=1)
        log.debug(f'[t={pipeline.t},n={pipeline.n}] Online - Snorm - Data_norm.shape   {pipeline.Data_norm.shape}')
        
        pipeline.processed_tr = norm_out