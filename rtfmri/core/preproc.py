import sys
import os.path as osp
import logging
import multiprocessing as mp

import numpy as np

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '../../../')))

from utils.core               import welford
from utils.rt_functions       import rt_EMA_vol, rt_regress_vol, rt_kalman_vol, kalman_filter_mv
from utils.rt_functions       import rt_smooth_vol, rt_snorm_vol, rt_svrscore_vol
from utils.rt_functions       import gen_polort_regressors

class Pipeline:
    def __init__(self, options, Nt, mask_Nv=None, mask_img=None, exp_type=None, log_level=None):
        self.Nt = Nt
        self.mask_Nv = mask_Nv
        self.mask_img = mask_img
        self.exp_type = exp_type

        self.processed_data = np.zeros((self.mask_Nv,1))

        self.logger = logging.getLogger('Preproc')
        self.logger.setLevel(logging.DEBUG) # TODO: fix this
        
        self.save_ema = options.save_ema
        self.save_smooth = options.save_smooth
        self.save_kalman = options.save_kalman
        self.save_iGLM = options.save_iglm
        self.save_orig = options.save_orig
        self.save_all = options.save_all

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
                self.logger.info(f'... initializing Kalman pool ...')
                _ = self.pool.map(kalman_filter_mv, self._initialize_kalman_pool())

    
    def _initialize_kalman_pool(self):
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
    

    def process_first_volume(self, t, n, this_t_data):
        self.Nv = len(this_t_data)
        """Create empty structures"""
        self.logger.info('Number of Voxels Nv=%d' % self.Nv)
        # if self.exp_type in ['esam', 'esam_test']:
        #     # These two variables are only needed if this is an experimental
        #     self.logger.debug(f'[t={t},n={n}] Initializing hits and svrscores')
        #     self..hits             = np.zeros((self..Ncaps, 1))
        #     self..svrscores        = np.zeros((self..Ncaps, 1))

        self.welford_M   = np.zeros(self.Nv)
        self.welford_S   = np.zeros(self.Nv)
        self.welford_std = np.zeros(self.Nv)
        if self.do_smooth:
            if self.mask_Nv != self.Nv:
                self.logger.error(f'Discrepancy across masks [data Nv = {self.Nv}, mask Nv = {self.mask_Nv}]')
                sys.exit(-1)
        self.Data_FromAFNI = np.array(this_t_data[:,np.newaxis])
        if self.save_ema:    self.Data_EMA      = np.zeros((self.Nv, 1))
        if self.save_iGLM:   self.Data_iGLM     = np.zeros((self.Nv, 1))
        if self.save_kalman: self.Data_kalman   = np.zeros((self.Nv, 1))
        if self.save_smooth: self.Data_smooth   = np.zeros((self.Nv, 1))
        self.Data_norm     = np.zeros((self.Nv,1))
        if self.save_iGLM:   self.iGLM_Coeffs   = np.zeros((self.Nv, self.iGLM_num_regressors, 1))
        self.S_x           = np.zeros(self.Nv)
        self.S_P           = np.zeros(self.Nv) 
        self.fPositDerivSpike = np.zeros(self.Nv)
        self.fNegatDerivSpike = np.zeros(self.Nv)
        if self.save_orig:   self.logger.debug(f'[t={t},n={n}] Init - Data_FromAFNI.shape {self.Data_FromAFNI.shape}')
        if self.save_ema:    self.logger.debug(f'[t={t},n={n}] Init - Data_EMA.shape      {self.Data_EMA.shape}') 
        if self.save_iGLM:   self.logger.debug(f'[t={t},n={n}] Init - Data_iGLM.shape     {self.Data_iGLM.shape}') 
        if self.save_kalman: self.logger.debug(f'[t={t},n={n}] Init - Data_kalman.shape   {self.Data_kalman.shape}')
        self.logger.debug(f'[t={t},n={n}] Init - Data_norm.shape     {self.Data_norm.shape}')
        if self.save_iGLM:   self.logger.debug(f'[t={t},n={n}] Init - iGLM_Coeffs.shape   {self.iGLM_Coeffs.shape}')
        return 1
    
    def process_discard(self, t, n, this_t_data):
        if self.save_orig: 
            self.Data_FromAFNI = np.append(self.Data_FromAFNI, this_t_data[:, np.newaxis], axis=1)
        else:
            self.Data_FromAFNI = np.hstack((self.Data_FromAFNI[:,-1][:,np.newaxis], this_t_data[:, np.newaxis]))  # Only keep this one and previous
        if self.save_ema:  self.Data_EMA      = np.append(self.Data_EMA,    np.zeros((self.Nv,1)), axis=1)
        if self.save_iGLM: self.Data_iGLM     = np.append(self.Data_iGLM,   np.zeros((self.Nv,1)), axis=1)
        if self.save_kalman: self.Data_kalman = np.append(self.Data_kalman, np.zeros((self.Nv,1)), axis=1)
        if self.save_smooth: self.Data_smooth = np.append(self.Data_smooth, np.zeros((self.Nv,1)), axis=1)
        self.Data_norm     = np.append(self.Data_norm,   np.zeros((self.Nv,1)), axis=1)
        if self.save_iGLM: self.iGLM_Coeffs   = np.append(self.iGLM_Coeffs, np.zeros( (self.Nv,self.iGLM_num_regressors,1)), axis=2)
        self.logger.debug('[t=%d,n=%d] Discard - Data_FromAFNI.shape %s' % (t, n, str(self.Data_FromAFNI.shape)))
        if self.save_ema:    self.logger.debug(f'[t={t},n={n}] Discard - Data_EMA.shape      {self.Data_EMA.shape}')
        if self.save_iGLM:   self.logger.debug(f'[t={t},n={n}] Discard - Data_iGLM.shape     {self.Data_iGLM.shape}')
        if self.save_kalman: self.logger.debug(f'[t={t},n={n}] Discard - Data_kalman.shape   {self.Data_kalman.shape}')
        self.logger.debug(f'[t={t},n={n}] Discard - Data_norm.shape    {self.Data_norm.shape}') 
        if self.save_iGLM: self.logger.debug(f'[t={t},n={n}] Discard - iGLM_Coeffs.shape   {self.iGLM_Coeffs.shape}')
        # if self.exp_type == "esam" or self.exp_type == "esam_test":
        #     # These two variables are only needed if this is an experimental
        #     self.hits      = np.append(self.hits,      np.zeros((self.Ncaps,1)),  axis=1)
        #     self.svrscores = np.append(self.svrscores, np.zeros((self.Ncaps,1)), axis=1)
        #     log.debug(f'[t={t},n={n}] Discard - hits.shape      %s' % (t, n, str(self.hits.shape)))
        #     log.debug(f'[t={t},n={n}] Discard - svrscores.shape %s' % (t, n, str(self.svrscores.shape)))
        
        self.logger.debug(f'Discard volume, self.Data_FromAFNI[:10]: {self.Data_FromAFNI[:10]}')
        return 1

    def run_welford(self, n, this_t_data):
        self.welford_M, self.welford_S, self.welford_std = welford(
            n,
            this_t_data,
            self.welford_M,
            self.welford_S
        )

        self.logger.debug(f'Welford Method Ouputs: M={self.welford_M} | S={self.welford_S} | std={self.welford_std}')
    
    def run_ema(self, t, n):
        ema_data_out, self.EMA_filt = rt_EMA_vol(n, self.EMA_th, self.Data_FromAFNI, self.EMA_filt, do_operation=self.do_EMA)
        if self.save_ema: 
            self.Data_EMA = np.append(self.Data_EMA, ema_data_out, axis=1)
            self.logger.debug(f'[t={t},n={n}] Online - EMA - Data_EMA.shape      {self.Data_EMA.shape}')
        
        self.processed_data = ema_data_out
    
    def run_iGLM(self, t, n, motion):
        if self.iGLM_motion:
            this_t_nuisance = np.concatenate((self.legendre_pols[t,:], motion))[:,np.newaxis]
        else:
            this_t_nuisance = (self.legendre_pols[t,:])[:,np.newaxis]
            
        iGLM_data_out, self.iGLM_prev, Bn = rt_regress_vol(
            n, 
            self.processed_data,
            this_t_nuisance,
            self.iGLM_prev,
            do_operation=self.do_iGLM
        )

        if self.save_iGLM: 
            self.Data_iGLM = np.append(self.Data_iGLM, iGLM_data_out, axis=1)
            self.iGLM_Coeffs = np.append(self.iGLM_Coeffs, Bn, axis=2) 
            self.logger.debug(f'[t={t},n={n}] Online - iGLM - Data_iGLM.shape     {self.Data_iGLM.shape}')
            self.logger.debug(f'[t={t},n={n}] Online - iGLM - iGLM_Coeffs.shape   {self.iGLM_Coeffs.shape}')

        self.processed_data = iGLM_data_out


    def run_kalman(self, t, n):
        klm_data_out, self.S_x, self.S_P, self.fPositDerivSpike, self.fNegatDerivSpike = rt_kalman_vol(
            n,
            t,
            self.processed_data,
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
            self.Data_kalman      = np.append(self.Data_kalman, klm_data_out, axis=1)
            self.logger.debug('[t=%d,n=%d] Online - Kalman - Data_kalman.shape     %s' % (t, n, str(self.Data_kalman.shape)))
        
        self.processed_data = np.squeeze(klm_data_out)

    def run_smooth(self, t, n):
        smooth_out = rt_smooth_vol(self.processed_data, self.mask_img, fwhm=self.FWHM, do_operation=self.do_smooth)
        if self.save_smooth:
            self.Data_smooth = np.append(self.Data_smooth, smooth_out, axis=1)
            self.logger.debug('[t=%d,n=%d] Online - Smooth - Data_smooth.shape   %s' % (t, n, str(self.Data_smooth.shape)))
            self.logger.debug('[t=%d,n=%d] Online - EMA - smooth_out.shape      %s' % (t, n, str(smooth_out.shape)))
            
        self.processed_data = np.squeeze(smooth_out)

    def run_snorm(self, t, n):
        # Should i make a self.save_norm? I'm pretty sure this doesn't exist becuase it's the last step, so it was just
        # whatever was saved in the end...
        norm_out = rt_snorm_vol(self.processed_data, do_operation=self.do_snorm)
    
        self.Data_norm = np.append(self.Data_norm, norm_out, axis=1)
        self.logger.debug(f'[t={t},n={n}] Online - Snorm - Data_norm.shape   {self.Data_norm.shape}')
        
        self.processed_data = self.Data_norm


    def process(self, t, n, motion, this_t_data):
        """Full preprocessing pipeline in original order"""        
        if t == 0:
            return self.process_first_volume(t, n, this_t_data)
        if n == 0:
            return self.process_discard(t, n, this_t_data)
        
        self.run_welford(n, this_t_data)

        # make this more flexible -- give users ability to define order in yaml (aka build Step class)
        if self.do_EMA:
            self.run_ema(t, n)
        if self.do_iGLM:
            self.run_iGLM(t, n, motion)
        if self.do_kalman:
            self.run_kalman(t, n)
        if self.do_smooth:
            self.run_smooth(t, n)
        if self.do_snorm:
            self.run_snorm(t, n)               

    def foo(self, t, n):
        iglm = Step('iGLM', self.run_iGLM, self.do_iGLM, [t, n])


class Step:
    """Not finished..."""
    def __init__(self, name, fn, do_fn, fn_args):
        self.name = name
        self.fn = fn
        self.do_fn = do_fn
        self.fn_args = fn_args
    
    def run(self):
        if self.do_fn:
            self.fn(*self.fn_args)
    
