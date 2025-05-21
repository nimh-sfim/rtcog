import sys
import os.path as osp
import multiprocessing as mp

import numpy as np
import pandas as pd

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '../../../')))

from utils.core               import welford
from utils.rt_functions       import rt_EMA_vol, rt_regress_vol, rt_kalman_vol, kalman_filter_mv
from utils.rt_functions       import rt_smooth_vol, rt_snorm_vol, rt_svrscore_vol
from utils.rt_functions       import gen_polort_regressors
from utils.log import get_logger
from utils.fMRI import load_fMRI_file, unmask_fMRI_img
from paths import OUTPUT_DIR, CAP_labels

log = get_logger()

DEFAULT_STEP_ORDER = [
    'EMA', 'iGLM', 'kalman', 'smooth', 'snorm'
]

class Pipeline:
    def __init__(self, options, Nt, mask_Nv=None, mask_img=None, exp_type=None):
        self.step_registry = {
            'EMA': 'run_ema',
            'iGLM': 'run_iGLM',
            'kalman': 'run_kalman',
            'smooth': 'run_smooth',
            'snorm': 'run_snorm',
        }
        self.build_steps()

        self.t = None
        self.n = None

        self.Nt = Nt
        self.mask_Nv = mask_Nv
        self.mask_img = mask_img
        self.exp_type = exp_type

        self.processed_tr = np.zeros((self.mask_Nv,1))

        self.motion_estimates = []
        
        self.save_ema = options.save_ema
        self.save_smooth = options.save_smooth
        self.save_kalman = options.save_kalman
        self.save_iGLM = options.save_iglm
        self.save_orig = options.save_orig
        self.save_all = options.save_all

        self.out_dir = options.out_dir
        self.out_prefix = options.out_prefix
        
        self.snapshot = options.snapshot
        if self.snapshot:
            self.save_all = True

        if self.save_all:
            self.save_orig = True
            self.save_ema = True
            self.save_iGLM = True
            self.save_kalman = True
            self.save_smooth = True

        self.welford_S = None
        self.welford_M = None
        self.welford_std = None

        self.Data_FromAFNI = None # np.array [Nv,Nt] for incoming data
        if self.save_ema:    self.Data_EMA = None # np.array [Nv,Nt] for data after EMA  step
        if self.save_iGLM:   self.Data_iGLM = None #np.array [Nv,Nt] for data after iGLM step
        if self.save_kalman: self.Data_kalman = None # np.array [Nv,Nt] for data after low-pass step
        if self.save_smooth: self.Data_smooth = None # np.array [Nv,Nt] for data after spatial smoothing
        if self.save_iGLM:   self.iGLM_Coeffs = None # np.array [Nregressor, Nv, Nt] for beta coefficients for all regressors
        self.Data_norm = None
        self.Data_processed = None

        # Preprocessing steps
        self.do_EMA = options.do_EMA
        self.do_iGLM = options.do_iGLM
        self.do_kalman = options.do_kalman
        self.do_smooth = options.do_smooth
        self.do_snorm = options.do_snorm

        self.FWHM = options.FWHM # FWHM for Spatial Smoothing in [mm]
        
        self.nvols_discard = options.discard      # Number of volumes to discard from any analysis (won't enter pre-processing)

        self.iGLM_prev     = {}
        self.iGLM_motion   = options.iGLM_motion
        self.iGLM_polort   = options.iGLM_polort
        self.nuisance      = None
        if self.iGLM_motion:
            self.iGLM_num_regressors = self.iGLM_polort + 6
            self.nuisance_labels = ['Polort'+str(i) for i in np.arange(self.iGLM_polort)] + ['roll','pitch','yaw','dS','dL','dP']
        else:
            self.iGLM_num_regressors = self.iGLM_polort
            self.nuisance_labels = ['Polort'+str(i) for i in np.arange(self.iGLM_polort)]

        self.S_x              = None
        self.S_P              = None
        self.fPositDerivSpike = None
        self.fNegatDerivSpike = None
        self.kalmThreshold    = None

        self.EMA_th   = 0.98
        self.EMA_filt = None

        # Create Legendre Polynomial regressors
        if self.iGLM_polort > -1:
            self.legendre_pols = gen_polort_regressors(self.iGLM_polort, self.Nt)
        else:
            self.legendre_pols = None

        # If kalman needed, create a pool
        if self.do_kalman:
            self.n_cores = options.n_cores
            self.pool = mp.Pool(processes=self.n_cores)
            if self.mask_Nv is not None:
                log.info(f'Initializing Kalman pool with {self.n_cores} processes ...')
                _ = self.pool.map(kalman_filter_mv, self._initialize_kalman_pool())
        else:
            self.n_cores = 0
            self.pool = None

    
    def _initialize_kalman_pool(self):
        """Initialize pool with fake data up front to avoid delay later"""
        Nv = int(self.mask_Nv)
        return [
            {
                'd': np.random.rand(Nv, 1),
                'std': np.random.rand(Nv),
                'S_x': np.zeros(Nv),
                'S_P': np.zeros(Nv),
                'S_Q': np.random.rand(Nv),
                'S_R': np.random.rand(Nv),
                'fPos': np.zeros(Nv),
                'fNeg': np.zeros(Nv),
                'vox': np.arange(Nv)
            }
            for _ in range(self.n_cores)
        ]
    
    def __del__(self):
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()
    
    def build_steps(self, step_order=None):
        # TODO: just skip adding the function if not do_function
        """Build the pipeline"""
        self.steps = []
        order = step_order or DEFAULT_STEP_ORDER
        for name in order:
            if name not in self.step_registry:
                log.error(f'Unknown step: {name}')
                sys.exit(-1)
            method = self.step_registry.get(name)
            self.steps.append(Step(name, method))

    def process_first_volume(self, this_t_data):
        """Create empty structures"""
        self.Nv = len(this_t_data)
        log.info('Number of Voxels Nv=%d' % self.Nv)

        self.welford_M   = np.zeros(self.Nv)
        self.welford_S   = np.zeros(self.Nv)
        self.welford_std = np.zeros(self.Nv)
        if self.do_smooth:
            if self.mask_Nv != self.Nv:
                log.error(f'Discrepancy across masks [data Nv = {self.Nv}, mask Nv = {self.mask_Nv}]')
                sys.exit(-1)
        self.Data_FromAFNI = np.array(this_t_data[:,np.newaxis])
        if self.save_ema:    self.Data_EMA      = np.zeros((self.Nv, 1))
        if self.save_iGLM:   self.Data_iGLM     = np.zeros((self.Nv, 1))
        if self.save_kalman: self.Data_kalman   = np.zeros((self.Nv, 1))
        if self.save_smooth: self.Data_smooth   = np.zeros((self.Nv, 1))
        self.Data_norm     = np.zeros((self.Nv,1))
        if self.save_iGLM:   self.iGLM_Coeffs   = np.zeros((self.Nv, self.iGLM_num_regressors, 1))
        self.Data_processed = np.zeros((self.Nv, 1)) # Final output
        
        self.S_x           = np.zeros(self.Nv)
        self.S_P           = np.zeros(self.Nv) 
        self.fPositDerivSpike = np.zeros(self.Nv)
        self.fNegatDerivSpike = np.zeros(self.Nv)
        if self.save_orig:   log.debug(f'[t={self.t},n={self.n}] Init - Data_FromAFNI.shape {self.Data_FromAFNI.shape}')
        if self.save_ema:    log.debug(f'[t={self.t},n={self.n}] Init - Data_EMA.shape      {self.Data_EMA.shape}') 
        if self.save_iGLM:   log.debug(f'[t={self.t},n={self.n}] Init - Data_iGLM.shape     {self.Data_iGLM.shape}') 
        if self.save_kalman: log.debug(f'[t={self.t},n={self.n}] Init - Data_kalman.shape   {self.Data_kalman.shape}')
        log.debug(f'[t={self.t},n={self.n}] Init - Data_norm.shape     {self.Data_norm.shape}')
        if self.save_iGLM:   log.debug(f'[t={self.t},n={self.n}] Init - iGLM_Coeffs.shape   {self.iGLM_Coeffs.shape}')
        return 1
    
    def process_discard(self, this_t_data):
        """Append a bunch of zeros for volumes we will be discarding"""
        if self.save_orig: 
            self.Data_FromAFNI = np.append(self.Data_FromAFNI, this_t_data[:, np.newaxis], axis=1)
        else:
            self.Data_FromAFNI = np.hstack((self.Data_FromAFNI[:,-1][:,np.newaxis], this_t_data[:, np.newaxis]))  # Only keep this one and previous
        if self.save_ema:  self.Data_EMA = np.append(self.Data_EMA,    np.zeros((self.Nv,1)), axis=1)
        if self.save_iGLM: self.Data_iGLM = np.append(self.Data_iGLM,   np.zeros((self.Nv,1)), axis=1)
        if self.save_kalman: self.Data_kalman = np.append(self.Data_kalman, np.zeros((self.Nv,1)), axis=1)
        if self.save_smooth: self.Data_smooth = np.append(self.Data_smooth, np.zeros((self.Nv,1)), axis=1)
        self.Data_norm = np.append(self.Data_norm, np.zeros((self.Nv,1)), axis=1)
        if self.save_iGLM: self.iGLM_Coeffs = np.append(self.iGLM_Coeffs, np.zeros( (self.Nv,self.iGLM_num_regressors,1)), axis=2)
        log.debug(f'[t={self.t},n={self.n}] Discard - Data_FromAFNI.shape {self.Data_FromAFNI.shape}')
        if self.save_ema:    log.debug(f'[t={self.t},n={self.n}] Discard - Data_EMA.shape      {self.Data_EMA.shape}')
        if self.save_iGLM:   log.debug(f'[t={self.t},n={self.n}] Discard - Data_iGLM.shape     {self.Data_iGLM.shape}')
        if self.save_kalman: log.debug(f'[t={self.t},n={self.n}] Discard - Data_kalman.shape   {self.Data_kalman.shape}')
        log.debug(f'[t={self.t},n={self.n}] Discard - Data_norm.shape    {self.Data_norm.shape}') 
        if self.save_iGLM: log.debug(f'[t={self.t},n={self.n}] Discard - iGLM_Coeffs.shape   {self.iGLM_Coeffs.shape}')
        self.Data_processed = np.append(self.Data_processed, np.zeros((self.Nv,1)), axis=1)
        
        log.debug(f'Discard volume, self.Data_FromAFNI[:10]: {self.Data_FromAFNI[:10]}')
        return 1

    def run_welford(self, this_t_data):
        self.welford_M, self.welford_S, self.welford_std = welford(
            self.n,
            this_t_data,
            self.welford_M,
            self.welford_S
        )

        log.debug(f'Welford Method Ouputs: M={self.welford_M} | S={self.welford_S} | std={self.welford_std}')
    
    def run_ema(self):
        log.debug(f'Before EMA: self.processed_tr.shape {self.processed_tr.shape}')
        ema_data_out, self.EMA_filt = rt_EMA_vol(self.n, self.EMA_th, self.Data_FromAFNI, self.EMA_filt, do_operation=self.do_EMA)
        if self.save_ema: 
            self.Data_EMA = np.append(self.Data_EMA, ema_data_out, axis=1)
            log.debug(f'[t={self.t},n={self.n}] Online - EMA - Data_EMA.shape      {self.Data_EMA.shape}')
        
        self.processed_tr = ema_data_out
        log.debug(f'After EMA: self.processed_tr.shape {self.processed_tr.shape}')
    
    def run_iGLM(self):
        log.debug(f'Before iGLM: self.processed_tr.shape {self.processed_tr.shape}')
        if self.iGLM_motion:
            this_t_nuisance = np.concatenate((self.legendre_pols[self.t,:], self.motion))[:,np.newaxis]
        else:
            this_t_nuisance = (self.legendre_pols[self.t,:])[:,np.newaxis]
            
        iGLM_data_out, self.iGLM_prev, Bn = rt_regress_vol(
            self.n, 
            self.processed_tr,
            this_t_nuisance,
            self.iGLM_prev,
            do_operation=self.do_iGLM
        )

        if self.save_iGLM: 
            self.Data_iGLM = np.append(self.Data_iGLM, iGLM_data_out, axis=1)
            self.iGLM_Coeffs = np.append(self.iGLM_Coeffs, Bn, axis=2) 
            log.debug(f'[t={self.t},n={self.n}] Online - iGLM - Data_iGLM.shape     {self.Data_iGLM.shape}')
            log.debug(f'[t={self.t},n={self.n}] Online - iGLM - iGLM_Coeffs.shape   {self.iGLM_Coeffs.shape}')

        self.processed_tr = iGLM_data_out
        log.debug(f'After iGLM: self.processed_tr.shape {self.processed_tr.shape}')

    def run_kalman(self):
        log.debug(f'Before kalman: self.processed_tr.shape {self.processed_tr.shape}')
        klm_data_out, self.S_x, self.S_P, self.fPositDerivSpike, self.fNegatDerivSpike = rt_kalman_vol(
            self.n,
            self.t,
            self.processed_tr,
            self.welford_std,
            self.S_x,
            self.S_P,
            self.fPositDerivSpike,
            self.fNegatDerivSpike,
            self.n_cores,
            self.pool,
            do_operation=self.do_kalman
        )
        
        if self.save_kalman: 
            self.Data_kalman = np.append(self.Data_kalman, klm_data_out, axis=1)
            log.debug(f'[t={self.t},n={self.n}] Online - Kalman - Data_kalman.shape     {self.Data_kalman.shape}')
        
        self.processed_tr = klm_data_out
        log.debug(f'After kalman: self.processed_tr.shape {self.processed_tr.shape}')

    def run_smooth(self):
        log.info(f'Before smooth: self.processed_tr.shape {self.processed_tr.shape}')
        smooth_out = rt_smooth_vol(self.processed_tr, self.mask_img, fwhm=self.FWHM, do_operation=self.do_smooth)
        if self.save_smooth:
            self.Data_smooth = np.append(self.Data_smooth, smooth_out, axis=1)
            log.debug(f'[t={self.t},n={self.n}] Online - Smooth - Data_smooth.shape   {self.Data_smooth.shape}')
            log.debug(f'[t={self.t},n={self.n}] Online - EMA - smooth_out.shape      {smooth_out.shape}')
            
        self.processed_tr = smooth_out
        log.debug(f'After smooth: self.processed_tr.shape {self.processed_tr.shape}')

    def run_snorm(self):
        log.debug(f'Before snorm: self.processed_tr.shape {self.processed_tr.shape}')
        # Should i make a self.save_norm? I'm pretty sure this doesn't exist becuase it's the last step, so it was just
        # whatever was saved in the end...
        norm_out = rt_snorm_vol(self.processed_tr, do_operation=self.do_snorm)
    
        self.Data_norm = np.append(self.Data_norm, norm_out, axis=1)
        log.debug(f'[t={self.t},n={self.n}] Online - Snorm - Data_norm.shape   {self.Data_norm.shape}')
        
        self.processed_tr = norm_out
        log.debug(f'After snorm: self.processed_tr.shape {self.processed_tr.shape}')


    def process(self, t, n, motion, this_t_data):
        """Full preprocessing pipeline in original order"""  
        self.t = t
        self.n = n
        self.motion = motion

        if self.t == 0:
            return self.process_first_volume(this_t_data)
        if self.n == 0:
            return self.process_discard(this_t_data)
        
        self.run_welford(this_t_data)
        
        if self.save_orig:
            self.Data_FromAFNI = np.append(self.Data_FromAFNI,this_t_data[:, np.newaxis], axis=1)
        else:
            self.Data_FromAFNI = np.hstack((self.Data_FromAFNI[:,-1][:,np.newaxis],this_t_data[:, np.newaxis]))  # Only keep this one and previous
            log.debug('[t=%d,n=%d] Online - Input - Data_FromAFNI.shape %s' % (self.t, self.n, str(self.Data_FromAFNI.shape)))

        for step in self.steps:
            step.run(self)     
        
        self.Data_processed = np.append(self.Data_processed, self.processed_tr, axis=1)
            

    def final_steps(self):
        self.save_motion_estimates()

        if self.mask_img is None:
            log.warning(' final_steps = No additional outputs generated due to lack of mask.')
            return 1
        
        log.debug(' final_steps - About to write outputs to disk.')
        self.save_nifti_files()

        # If running snapshot test, save the variable states
        if self.snapshot:
            var_dict = {
                'Data_norm': self.Data_norm,
                'Data_EMA': self.Data_EMA,
                'Data_iGLM': self.Data_iGLM,
                'Data_smooth': self.Data_smooth,
                # 'Data_kalman': self.Data_kalman,
                'Data_FromAFNI': self.Data_FromAFNI,
                'Data_processed': self.Data_processed
            }
        
            snap_path = osp.join(OUTPUT_DIR, f'{self.out_prefix}_snapshots.npz')
            np.savez(snap_path, **var_dict) 
            log.info(f'Snapshot saved to OUTPUT_DIR at {snap_path}')
        
    def save_motion_estimates(self):
        self.motion_estimates = [item for sublist in self.motion_estimates for item in sublist]
        log.info(f'motion_estimates length is {len(self.motion_estimates)}')
        self.motion_estimates = np.reshape(self.motion_estimates,newshape=(int(len(self.motion_estimates)/6),6))
        motion_path = osp.join(self.out_dir,self.out_prefix+'.Motion.1D')
        np.savetxt(
            motion_path, 
            self.motion_estimates,
            delimiter="\t"
        )
        log.info(f'Motion estimates saved to disk: [{motion_path}]')

    def save_nifti_files(self):
        out_vars   = [self.Data_processed]
        out_labels = ['.pp_Zscore.nii']
        if self.do_EMA and self.save_ema:
            out_vars.append(self.Data_EMA)
            out_labels.append('.pp_EMA.nii')
        if self.do_iGLM and self.save_iGLM:
            out_vars.append(self.Data_iGLM)
            out_labels.append('.pp_iGLM.nii')
        if self.do_kalman and self.save_kalman:
            out_vars.append(self.Data_kalman)
            out_labels.append('.pp_LPfilter.nii')
        if self.do_smooth and self.save_smooth:
            out_vars.append(self.Data_smooth)
            out_labels.append('.pp_Smooth.nii')
        if self.save_orig:
            out_vars.append(self.Data_FromAFNI)
            out_labels.append('.orig.nii')
        
        for variable, file_suffix in zip(out_vars, out_labels):
            unmask_fMRI_img(variable, self.mask_img, osp.join(self.out_dir,self.out_prefix+file_suffix))

        if self.do_iGLM and self.save_iGLM:
            for i,lab in enumerate(self.nuisance_labels):
                data = self.iGLM_Coeffs[:,i,:]
                unmask_fMRI_img(data, self.mask_img, osp.join(self.out_dir,self.out_prefix+'.pp_iGLM_'+lab+'.nii'))

                                   
class Step:
    """Preprocessing step.
    TODO:
        - pass method itself so that users can easily create their own
    """
    def __init__(self, name, method_name):
        self.name = name
        self.method_name = method_name

    def run(self, pipeline):
        fn = getattr(pipeline, self.method_name)
        fn()

