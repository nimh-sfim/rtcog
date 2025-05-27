import numpy as np

from utils.rt_functions       import rt_EMA_vol, rt_regress_vol, rt_kalman_vol, rt_smooth_vol, rt_snorm_vol
from utils.log import get_logger

log = get_logger()

class PreprocStep:
    """
    Base class for preprocessing steps in the real-time fMRI pipeline.

    Each subclass must implement a `run(pipeline)` method, which:
      - Reads `pipeline.processed_tr` (a numpy array of shape (N_voxels, 1))
      - Performs its step-specific transformation
      - Updates `pipeline.processed_tr` with a new numpy array of the same shape

    Limitations:
      - Input/output data shape must be (N_voxels, 1).
      - Will raise runtime errors if this condition is not met.
    """
    @property
    def name(self):
        return self.__class__.__name__
    
    def run(self, pipeline):
        raise NotImplementedError
    
class EMAStep(PreprocStep):
    def run(self, pipeline):
        ema_data_out, pipeline.EMA_filt = rt_EMA_vol(pipeline.n, pipeline.EMA_th, pipeline.Data_FromAFNI, pipeline.EMA_filt)
        if pipeline.save_ema: 
            pipeline.Data_EMA = np.append(pipeline.Data_EMA, ema_data_out, axis=1)
            log.debug(f'[t={pipeline.t},n={pipeline.n}] Online - EMA - Data_EMA.shape      {pipeline.Data_EMA.shape}')
        
        pipeline.processed_tr = ema_data_out
    
class iGLMStep(PreprocStep):
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
    def run(self, pipeline):
        smooth_out = rt_smooth_vol(pipeline.processed_tr, pipeline.mask_img, fwhm=pipeline.FWHM)
        if pipeline.save_smooth:
            pipeline.Data_smooth = np.append(pipeline.Data_smooth, smooth_out, axis=1)
            log.debug(f'[t={pipeline.t},n={pipeline.n}] Online - Smooth - Data_smooth.shape   {pipeline.Data_smooth.shape}')
            log.debug(f'[t={pipeline.t},n={pipeline.n}] Online - EMA - smooth_out.shape      {smooth_out.shape}')
            
        pipeline.processed_tr = smooth_out

class SnormStep(PreprocStep):
    def run(self, pipeline):
        # Should i make a pipeline.save_norm? I'm pretty sure this doesn't exist becuase it's the last step, so it was just
        # whatever was saved in the end...
        norm_out = rt_snorm_vol(pipeline.processed_tr)
    
        pipeline.Data_norm = np.append(pipeline.Data_norm, norm_out, axis=1)
        log.debug(f'[t={pipeline.t},n={pipeline.n}] Online - Snorm - Data_norm.shape   {pipeline.Data_norm.shape}')
        
        pipeline.processed_tr = norm_out

DEFAULT_STEP_ORDER = [
    'EMA', 'iGLM', 'kalman', 'smooth', 'snorm'
]

STEP_REGISTRY = {
    'iGLM':    (iGLMStep, lambda p: p.do_iGLM),
    'kalman':  (KalmanStep, lambda p: p.do_kalman),
    'smooth':  (SmoothStep, lambda p: p.do_smooth),
    'snorm':   (SnormStep, lambda p: p.do_snorm),
    'EMA':     (EMAStep, lambda p: p.do_EMA),
}