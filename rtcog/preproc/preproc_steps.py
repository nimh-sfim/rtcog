import multiprocessing as mp
import os.path as osp
import numpy as np

from rtcog.utils.log import get_logger
from rtcog.utils.exceptions import VolumeOverflowError
from rtcog.preproc.preproc_utils import gen_polort_regressors
from rtcog.preproc.preproc_utils import initialize_kalman_pool, kalman_filter_mv, rt_kalman_vol
from rtcog.preproc.preproc_utils import rt_EMA_vol, rt_regress_vol, rt_smooth_vol, rt_snorm_vol, welford, calculate_spc
from rtcog.preproc.preproc_utils import create_win, CircularBuffer
from rtcog.preproc.step_types import StepType
from rtcog.utils.fMRI import unmask_fMRI_img


log = get_logger()

class PreprocStep:
    """
    Base class for a preprocessing step in the real-time fMRI pipeline.

    Subclasses must implement the `_run(pipeline)` method, which is called on each TR 
    and receives access to the pipeline’s state. Optionally, subclasses may also implement 
    `_start(pipeline)` to initialize state at the first TR, and `_save(pipeline)` to 
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
    end_step(pipeline):
        Optional cleanup logic to perform after processing has completed.
    save_nifti(pipeline):
        Saves the accumulated data to disk as a NIfTI file using the provided mask and filename.
    get_class(name):
        Class method that retrieves a registered step class by name (case-insensitive).
    """
    registry = {} # Holds all available step classes that can be instantiated later on in Pipeline.

    def __init__(self, save=False, suffix=None, **kwargs):
        self.save = save
        self.suffix = suffix

        self.data_out = None

    @property
    def name(self):
        return self.__class__.__name__.replace('Step', '').lower()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Skip abstract or helper base classes
        if cls.__name__ == "PreprocStep" or cls.__name__.startswith("_"):
            return
    
        # Strip "Step" from the end of class names
        name = cls.__name__
        if name.endswith("Step"):
            name = name[:-4]
        name = name.lower()

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
    
    def end_step(self, pipeline):
        pass

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
    """Exponential moving average"""
    def __init__(self, save, suffix='.pp_EMA.nii', ema_thr=0.98):
        super().__init__(save, suffix)
        self.EMA_thr = ema_thr
        self.EMA_filt = None

    def _run(self, pipeline):
        Data_FromAFNI = pipeline.Data_FromAFNI[:, :pipeline.t + 1]
        ema_data_out, self.EMA_filt = rt_EMA_vol(pipeline.n, self.EMA_thr, Data_FromAFNI, self.EMA_filt)
        
        return ema_data_out
    

class iGLMStep(PreprocStep):
    """Incremental generalized linear model"""
    def __init__(self, save, suffix='.pp_iGLM.nii', num_polorts=2, iGLM_motion=True):
        super().__init__(save, suffix)
        self.num_polorts = num_polorts
        self.iGLM_motion = iGLM_motion

        self.iGLM_prev = {}

        if self.save:
            self.iGLM_Coeffs = None
        
    def _start(self, pipeline):
        if self.iGLM_motion:
            self.iGLM_num_regressors = self.num_polorts + 6
            self.nuisance_labels = ['Polort'+str(i) for i in np.arange(self.num_polorts)] + ['roll','pitch','yaw','dS','dL','dP']
        else:
            self.iGLM_num_regressors = self.num_polorts
            self.nuisance_labels = ['Polort'+str(i) for i in np.arange(self.num_polorts)]
        
        if self.num_polorts > -1:
            self.legendre_pols = gen_polort_regressors(self.num_polorts, pipeline.Nt)
        else:
            self.legendre_pols = None

        if self.save:
            self.iGLM_Coeffs = np.zeros((pipeline.Nv, self.iGLM_num_regressors, pipeline.Nt))

    def _run(self, pipeline): 
        try:
            if self.iGLM_motion:
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
    """Kalman filter for low pass filtering, spike removal, and signal smoothing"""
    def __init__(self, save, suffix='.pp_LPfilter.nii', n_cores=10, mask_Nv=None):
        super().__init__(save, suffix)
        self.welford_S = None
        self.welford_M = None
        self.welford_std = None

        self.S_x = None
        self.S_P = None
        self.fPositDerivSpike = None
        self.fNegatDerivSpike = None
        
        self.n_cores = n_cores
        self.pool = mp.Pool(processes=self.n_cores)
        self.mask_Nv = mask_Nv

        log.info(f'Initializing Kalman pool with {self.n_cores} processes ...')
        _ = self.pool.map(kalman_filter_mv, initialize_kalman_pool(self.mask_Nv, self.n_cores))        

    def _start(self, pipeline):
        Nv = pipeline.Nv
        self.welford_M = np.zeros(Nv)
        self.welford_S   = np.zeros(Nv)
        self.welford_std = np.zeros(Nv)

        self.S_x = np.zeros(Nv)
        self.S_P = np.zeros(Nv) 
        self.fPositDerivSpike = np.zeros(Nv)
        self.fNegatDerivSpike = np.zeros(Nv)

    def _run(self, pipeline):
        self.welford_M, self.welford_S, self.welford_std = welford(
            pipeline.n,
            pipeline.Data_FromAFNI[:, pipeline.t],
            self.welford_M,
            self.welford_S
        )

        klm_data_out, S_x_new, S_P_new, fPos_new, fNeg_new = rt_kalman_vol(
            pipeline.n,
            pipeline.t,
            pipeline.processed_tr,
            self.welford_std,
            self.S_x,
            self.S_P,
            self.fPositDerivSpike,
            self.fNegatDerivSpike,
            self.n_cores,
            self.pool,
        )

        self.S_x[:] = S_x_new.ravel()
        self.S_P[:] = S_P_new.ravel()
        self.fPositDerivSpike[:] = fPos_new.ravel()
        self.fNegatDerivSpike[:] = fNeg_new.ravel()

        return klm_data_out

    def end_step(self, pipeline):
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.join()
            del self.pool


class SmoothStep(PreprocStep):
    """Smoothing with Gaussian filter"""
    def __init__(self, save=False, suffix='.pp_Smooth.nii', fwhm=4):
        super().__init__(save, suffix)
        self.fwhm = fwhm

    def _run(self, pipeline):
        smooth_out = rt_smooth_vol(pipeline.processed_tr, pipeline.mask_img, fwhm=self.fwhm)
            
        return smooth_out

    
class SnormStep(PreprocStep):
    """Spatial normalization"""
    def __init__(self, save=False, suffix='.pp_Zscore.nii'):
        super().__init__(save, suffix)

    def _run(self, pipeline):
        norm_out = rt_snorm_vol(pipeline.processed_tr)
        
        return norm_out

                
class TnormStep(PreprocStep):
    """Temporal normalization"""
    def __init__(self, save=False, suffix='.pp_Tnorm.nii', nvols_to_compute=50):
        super().__init__(save, suffix)
        self.nvols_to_compute = nvols_to_compute
        self.mean_removed = False
        self.fwhm = None

        self.orig_data = None
        self.baseline_signal = None

    def _start(self, pipeline):
        self.orig_data = np.zeros((pipeline.Nv, self.nvols_to_compute))

        smooth_step = next((step for step in pipeline.steps if step.name == StepType.SMOOTH.value), None)
        if smooth_step:
            self.fwhm = smooth_step.fwhm
       
        #TODO: determine if iGLM presence with num_polorts >= 1 should also count here
        self.mean_removed = StepType.EMA.value in pipeline.steps

    def _run(self, pipeline):
        n = pipeline.n
        t = pipeline.t

        # Calculate SPC after baseline is established
        if self.baseline_signal is not None:
            current_signal = pipeline.processed_tr[:, 0]
            return calculate_spc(current_signal, self.baseline_signal, self.mean_removed)

        # Collect orig data until nvols_to_compute is reached
        elif 0 < n <= self.nvols_to_compute:
            vol = pipeline.Data_FromAFNI[:, pipeline.t]

            if self.fwhm is not None: # Smooth if necessary
                vol = rt_smooth_vol(vol[:, np.newaxis], pipeline.mask_img, fwhm=self.fwhm)
            
            self.orig_data[:, n - 1] = vol[:, 0]
            
            # Establish baseline
            if n == self.nvols_to_compute:
                print(f"++ INFO: establishing baseline at {t=}")
                self.baseline_signal = np.mean(self.orig_data, axis=1)
                self.baseline_signal[self.baseline_signal == 0] = 1e-6 # Avoid divide by zero
                print(f'{self.baseline_signal.shape=}')

            return pipeline.processed_tr

        # During discard volumes
        return pipeline.processed_tr
        

class WindowingStep(PreprocStep):
    def __init__(self, save=False, suffix='.pp_Windowed.nii', win_length=4):
        super().__init__(save, suffix)
        self.buffer = None

        self.win_length = win_length
        self.hit_win_weights = create_win(self.win_length)
    
    # NOTE: windowing used to only be done once matching began, now starting it along with all other preproc steps.
    def _run(self, pipeline):
        if self.buffer is None:
            self.buffer = CircularBuffer(pipeline.processed_tr.shape[0], self.win_length)
        current_window = self.buffer.update(pipeline.processed_tr)
        if current_window is not None:
            return np.dot(current_window, self.hit_win_weights)

        return pipeline.processed_tr
        
    