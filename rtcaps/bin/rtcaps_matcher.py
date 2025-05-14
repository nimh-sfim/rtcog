import sys
import time
import argparse
import logging
import json
import pickle
import os.path as osp
import multiprocessing as mp 
from time import sleep
from psychopy import event
import holoviews as hv
import hvplot.pandas
import numpy as np
import pandas as pd

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))

from config                       import RESOURCES_DIR, CAP_labels
# from rtcap_lib.receiver_interface import ReceiverInterface
from rtcap_lib.receiver_interface import CustomReceiverInterface
from rtcap_lib.core               import welford
from rtcap_lib.rt_functions       import rt_EMA_vol, rt_regress_vol, rt_kalman_vol, kalman_filter_mv
from rtcap_lib.rt_functions       import rt_smooth_vol, rt_snorm_vol, rt_svrscore_vol
from rtcap_lib.rt_functions       import gen_polort_regressors
from rtcap_lib.fMRI               import load_fMRI_file, unmask_fMRI_img
from rtcap_lib.svr_methods        import is_hit_rt01
from rtcap_lib.core               import create_win
from rtcap_lib.experiment_qa      import get_experiment_info, DefaultScreen, QAScreen

log = logging.getLogger('online_preproc')

# if not log.hasHandlers():
    # print('++ LOGGER: setting')
log.setLevel(logging.INFO)

log_fmt = logging.Formatter('[%(levelname)s - %(filename)s]: %(message)s')

# File Handler (overwriting the log each time)
file_handler = logging.FileHandler('online_preproc.log', mode='w')
file_handler.setFormatter(log_fmt)

# Stream Handler (for console output)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_fmt)

# Add handlers to the logger
log.addHandler(file_handler)
log.addHandler(stream_handler)
# print(f'++ LOGGER HANDLERS: {log.handlers}')

from psychopy import prefs
prefs.hardware['audioLib'] = ['pyo']
prefs.hardware['keyboard'] = 'pygame'
prefs.hardware['audio'] = 'pygame'

# Ignore psychopy warnings
from psychopy import logging
logging.console.setLevel(logging.ERROR)

class Experiment:
    def __init__(self, options, mp_evt_hit, mp_evt_end, mp_evt_qa_end):

        self.mp_evt_hit = mp_evt_hit           # Signals a CAP hit
        self.mp_evt_end = mp_evt_end           # Signals the end of the experiment
        self.mp_evt_qa_end = mp_evt_qa_end     # Signals the end of a QA set
        self.silent        = options.silent
        self.debug         = options.debug
        if self.debug:
            log.setLevel(logging.DEBUG)
        if self.silent:
            log.setLevel(logging.CRITICAL)

        self.ewin          = None
        self.exp_type      = options.exp_type
        self.no_proc_chair = options.no_proc_chair
        self.screen_size   = [512, 288]
        self.fullscreen    = options.fullscreen
        self.screen        = options.screen

        self.n             = 0               # Counter for number of volumes pre-processed (Start = 1)
        self.t             = -1              # Counter for number of received volumes (Start = 0)
        self.lastQA_endTR  = 0
        self.vols_noqa     = options.vols_noqa
        self.Nv            = None            # Number of voxels in data mask
        self.Nt            = options.nvols   # Number acquisitions
        self.TR            = options.tr      # TR [seconds]
        self.n_cores       = options.n_cores # Number of cores for the multiprocessing part of the code
        self.save_ema      = options.save_ema
        self.save_smooth   = options.save_smooth
        self.save_kalman   = options.save_kalman
        self.save_iGLM     = options.save_iglm
        self.save_orig     = options.save_orig
        self.save_all      = options.save_all
        
        if self.save_all:
            self.save_orig   = True
            self.save_ema    = True
            self.save_iGLM   = True
            self.save_kalman = True
            self.save_smooth = True

        self.welford_S     = None
        self.welford_M     = None
        self.welford_std   = None
        self.Data_FromAFNI = None            # np.array [Nv,Nt] for incoming data
        if self.save_ema:    self.Data_EMA      = None            # np.array [Nv,Nt] for data after EMA  step
        if self.save_iGLM:   self.Data_iGLM     = None            # np.array [Nv,Nt] for data after iGLM step
        if self.save_kalman: self.Data_kalman   = None            # np.array [Nv,Nt] for data after low-pass step
        if self.save_smooth: self.Data_smooth   = None            # np.array [Nv,Nt] for data after spatial smoothing
        if self.save_iGLM:   self.iGLM_Coeffs   = None            # np.array [Nregressor, Nv, Nt] for beta coefficients for all regressors
        self.Data_norm     = None            # np.array [Nv,Nt] for data after spatial normalization (spatial z-score)

        self.do_EMA        = options.do_EMA       # Should we do EMA
        self.do_iGLM       = options.do_iGLM      # Should we do iGLM
        self.do_kalman     = options.do_kalman    # Should we do Low Pass Filter
        self.do_smooth     = options.do_smooth    # Should we do Spatial Filtering
        self.do_snorm      = options.do_snorm     # Should we do Spatial Z-scoring per volume
        self.FWHM          = options.FWHM         # FWHM for Spatial Smoothing in [mm]
        
        self.nvols_discard = options.discard      # Number of volumes to discard from any analysis (won't enter pre-processing)

        self.iGLM_prev     = {}
        self.iGLM_motion   = options.iGLM_motion
        self.iGLM_polort   = options.iGLM_polort
        self.nuisance      = None
        self.motion_estimates = []
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

        self.mask_path  = options.mask_path
        self.out_dir    = options.out_dir
        self.out_prefix = options.out_prefix 
        self.outhtml       = osp.join(self.out_dir,self.out_prefix+'.dyn_report')

        self.qa_onsets  = []
        self.qa_offsets = []
        self.qa_onsets_path  = osp.join(self.out_dir,self.out_prefix+'.qa_onsets.txt')
        self.qa_offsets_path = osp.join(self.out_dir,self.out_prefix+'.qa_offsets.txt')
        # Load Mask - Necessary for smoothing
        if self.do_smooth and self.mask_path is None:
            log.error('   Experiment_init_ - Smoothing requires a mask. Provide a mask or disable smoothing operation.')
            sys.exit(-1)
        
        if self.mask_path is None:
            log.warning('  Experiment_init_ - No mask was provided!')
            self.mask_img = None
        else:
            self.mask_img  = load_fMRI_file(self.mask_path)
            self.mask_Nv = np.sum(self.mask_img.get_fdata())
            log.debug('  Experiment_init_ - Number of Voxels in user-provided mask: %d' % self.mask_Nv)

        # Create Legendre Polynomial regressors
        if self.iGLM_polort > -1:
            self.legendre_pols = gen_polort_regressors(self.iGLM_polort,self.Nt)
        else:
            self.legendre_pols = None

        # If kalman needed, create a pool
        if self.do_kalman:
            self.pool = mp.Pool(processes=self.n_cores)
            if self.mask_Nv is not None:
                log.info(f'   Experiment_init_ - Initializing Kalman pool with {self.n_cores} worker processes using dummy data.')
                _ = self.pool.map(kalman_filter_mv, self._initialize_kalman_pool())
        else:
            self.pool = None

        # For snapshot testing
        self.snapshot = options.snapshot


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


    def compute_TR_data(self, motion, extra):
        # NOTE: extra and save_orig's data_FromAFNI are identical
        # Status as we enter the function
        hit_status    = self.mp_evt_hit.is_set()
        qa_end_status = self.mp_evt_qa_end.is_set()

        # Update t up front
        self.t += 1

        # Keep a record of motion estimates
        motion = [i[self.t] for i in motion]
        self.motion_estimates.append(motion)
        if len(motion) != 6:
            log.error('Motion not read in correctly.')
            log.error(f'Expected length: 6 | Actual length: {len(motion)}')
            sys.exit(-1)
        
        this_t_data = np.array([e[self.t] for e in extra])
        if self.t > 0:
            if len(this_t_data) != self.Nv:
                log.error(f'Extra data not read in correctly.')
                log.error(f'Expected length: {self.Nv} | Actual length: {len(this_t_data)}')
                sys.exit(-1)

        del extra # Save resources

        # Update n (only if not longer a discard volume)
        if self.t > self.nvols_discard - 1:
            self.n += 1

        log.info(' - Time point [t=%d, n=%d] | lastQA_endTR = %d | hit = %s | qa_end = %s | prg_end = %s' % (self.t, self.n, self.lastQA_endTR,
                                    self.mp_evt_hit.is_set(),
                                    self.mp_evt_qa_end.is_set(),
                                    self.mp_evt_end.is_set()))
        
        # If first volume, then create empty structures and call it a day (TR)
        if self.t == 0:
            self.Nv = len(this_t_data)
            log.info('Number of Voxels Nv=%d' % self.Nv)
            if self.exp_type == "esam" or self.exp_type == "esam_test":
                # These two variables are only needed if this is an experimental
                log.debug('[t=%d,n=%d] Initializing hits and svrscores' % (self.t, self.n))
                self.hits             = np.zeros((self.Ncaps, 1))
                self.svrscores        = np.zeros((self.Ncaps, 1))

            self.welford_M   = np.zeros(self.Nv)
            self.welford_S   = np.zeros(self.Nv)
            self.welford_std = np.zeros(self.Nv)
            if self.do_smooth:
                if self.mask_Nv != self.Nv:
                    log.error('Discrepancy across masks [data Nv = %d, mask Nv = %d]' % (self.Nv, self.mask_Nv) )
                    sys.exit(-1)
            self.Data_FromAFNI = np.array(this_t_data[:,np.newaxis])
            if self.save_ema:    self.Data_EMA      = np.zeros((self.Nv,1))
            if self.save_iGLM:   self.Data_iGLM     = np.zeros((self.Nv,1))
            if self.save_kalman: self.Data_kalman   = np.zeros((self.Nv,1))
            if self.save_smooth: self.Data_smooth   = np.zeros((self.Nv,1))
            self.Data_norm     = np.zeros((self.Nv,1))
            if self.save_iGLM:   self.iGLM_Coeffs   = np.zeros((self.Nv,self.iGLM_num_regressors,1))
            self.S_x           = np.zeros(self.Nv)
            self.S_P           = np.zeros(self.Nv) 
            self.fPositDerivSpike = np.zeros(self.Nv)
            self.fNegatDerivSpike = np.zeros(self.Nv)
            if self.save_orig:   log.debug('[t=%d,n=%d] Init - Data_FromAFNI.shape %s' % (self.t, self.n, str(self.Data_FromAFNI.shape)))
            if self.save_ema:    log.debug('[t=%d,n=%d] Init - Data_EMA.shape      %s' % (self.t, self.n, str(self.Data_EMA.shape)))
            if self.save_iGLM:   log.debug('[t=%d,n=%d] Init - Data_iGLM.shape     %s' % (self.t, self.n, str(self.Data_iGLM.shape)))
            if self.save_kalman: log.debug('[t=%d,n=%d] Init - Data_kalman.shape   %s' % (self.t, self.n, str(self.Data_kalman.shape)))
            log.debug('[t=%d,n=%d] Init - Data_norm.shape     %s' % (self.t, self.n, str(self.Data_norm.shape)))
            if self.save_iGLM: log.debug('[t=%d,n=%d] Init - iGLM_Coeffs.shape   %s' % (self.t, self.n, str(self.iGLM_Coeffs.shape)))
            return 1

        # For any other vol, if still a discard volume
        if self.n == 0:
            if self.save_orig: 
                self.Data_FromAFNI = np.append(self.Data_FromAFNI,this_t_data[:, np.newaxis], axis=1)
            else:
                self.Data_FromAFNI = np.hstack((self.Data_FromAFNI[:,-1][:,np.newaxis],this_t_data[:, np.newaxis]))  # Only keep this one and previous
            if self.save_ema:  self.Data_EMA      = np.append(self.Data_EMA,    np.zeros((self.Nv,1)), axis=1)
            if self.save_iGLM: self.Data_iGLM     = np.append(self.Data_iGLM,   np.zeros((self.Nv,1)), axis=1)
            if self.save_kalman: self.Data_kalman = np.append(self.Data_kalman, np.zeros((self.Nv,1)), axis=1)
            if self.save_smooth: self.Data_smooth = np.append(self.Data_smooth, np.zeros((self.Nv,1)), axis=1)
            self.Data_norm     = np.append(self.Data_norm,   np.zeros((self.Nv,1)), axis=1)
            if self.save_iGLM: self.iGLM_Coeffs   = np.append(self.iGLM_Coeffs, np.zeros( (self.Nv,self.iGLM_num_regressors,1)), axis=2)
            log.debug('[t=%d,n=%d] Discard - Data_FromAFNI.shape %s' % (self.t, self.n, str(self.Data_FromAFNI.shape)))
            if self.save_ema:    log.debug('[t=%d,n=%d] Discard - Data_EMA.shape      %s' % (self.t, self.n, str(self.Data_EMA.shape)))
            if self.save_iGLM:   log.debug('[t=%d,n=%d] Discard - Data_iGLM.shape     %s' % (self.t, self.n, str(self.Data_iGLM.shape)))
            if self.save_kalman: log.debug('[t=%d,n=%d] Discard - Data_kalman.shape   %s' % (self.t, self.n, str(self.Data_kalman.shape)))
            log.debug('[t=%d,n=%d] Discard - Data_norm.shape     %s' % (self.t, self.n, str(self.Data_norm.shape)))
            if self.save_iGLM: log.debug('[t=%d,n=%d] Discard - iGLM_Coeffs.shape   %s' % (self.t, self.n, str(self.iGLM_Coeffs.shape)))
            if self.exp_type == "esam" or self.exp_type == "esam_test":
                # These two variables are only needed if this is an experimental
                self.hits      = np.append(self.hits,      np.zeros((self.Ncaps,1)),  axis=1)
                self.svrscores = np.append(self.svrscores, np.zeros((self.Ncaps,1)), axis=1)
                log.debug('[t=%d,n=%d] Discard - hits.shape      %s' % (self.t, self.n, str(self.hits.shape)))
                log.debug('[t=%d,n=%d] Discard - svrscores.shape %s' % (self.t, self.n, str(self.svrscores.shape)))
            
            log.debug(f'Discard volume, self.Data_FromAFNI[:10]: {self.Data_FromAFNI[:10]}')
            return 1

        # Compute running mean and running std with welford
        self.welford_M, self.welford_S, self.welford_std = welford(self.n, this_t_data, self.welford_M, self.welford_S)
        log.debug('Welford Method Ouputs: M=%s | S=%s | std=%s' % (str(self.welford_M), str(self.welford_S), str(self.welford_std)))
        
        # If we reach this point, it means we have work to do
        if self.save_orig:
            self.Data_FromAFNI = np.append(self.Data_FromAFNI,this_t_data[:, np.newaxis], axis=1)
        else:
            self.Data_FromAFNI = np.hstack((self.Data_FromAFNI[:,-1][:,np.newaxis],this_t_data[:, np.newaxis]))  # Only keep this one and previous
            log.debug('[t=%d,n=%d] Online - Input - Data_FromAFNI.shape %s' % (self.t, self.n, str(self.Data_FromAFNI.shape)))
        
        
        # Do EMA (if needed)
        # =================
        ema_data_out, self.EMA_filt = rt_EMA_vol(self.n, self.EMA_th, self.Data_FromAFNI, self.EMA_filt, do_operation=self.do_EMA)
        if self.save_ema: 
            self.Data_EMA = np.append(self.Data_EMA, ema_data_out, axis=1)
            log.debug('[t=%d,n=%d] Online - EMA - Data_EMA.shape      %s' % (self.t, self.n, str(self.Data_EMA.shape)))
        # Do iGLM (if needed)
        # ===================
        if self.iGLM_motion:
            this_t_nuisance = np.concatenate((self.legendre_pols[self.t,:],motion))[:,np.newaxis]
        else:
            this_t_nuisance = (self.legendre_pols[self.t,:])[:,np.newaxis]
            
        iGLM_data_out, self.iGLM_prev, Bn = rt_regress_vol(
            self.n, 
            ema_data_out,
            this_t_nuisance,
            self.iGLM_prev,
            do_operation=self.do_iGLM
        )
        
        if self.save_iGLM: 
            self.Data_iGLM    = np.append(self.Data_iGLM, iGLM_data_out, axis=1)
            self.iGLM_Coeffs  = np.append(self.iGLM_Coeffs, Bn, axis = 2) 
            log.debug('[t=%d,n=%d] Online - iGLM - Data_iGLM.shape     %s' % (self.t, self.n, str(self.Data_iGLM.shape)))
            log.debug('[t=%d,n=%d] Online - iGLM - iGLM_Coeffs.shape   %s' % (self.t, self.n, str(self.iGLM_Coeffs.shape)))
        # Do Kalman Low-Pass Filter (if needed)
        # =====================================
        klm_data_out, self.S_x, self.S_P, self.fPositDerivSpike, self.fNegatDerivSpike = rt_kalman_vol(
            self.n,
            self.t,
            iGLM_data_out,
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
            log.debug('[t=%d,n=%d] Online - Kalman - Data_kalman.shape     %s' % (self.t, self.n, str(self.Data_kalman.shape)))
        # Do Spatial Smoothing (if needed)
        # ================================
        smooth_out = rt_smooth_vol(np.squeeze(klm_data_out), self.mask_img, fwhm=self.FWHM, do_operation=self.do_smooth)
        if self.save_smooth:
            self.Data_smooth = np.append(self.Data_smooth, smooth_out, axis=1)
            log.debug('[t=%d,n=%d] Online - Smooth - Data_smooth.shape   %s' % (self.t, self.n, str(self.Data_smooth.shape)))
            log.debug('[t=%d,n=%d] Online - EMA - smooth_out.shape      %s' % (self.t, self.n, str(smooth_out.shape)))
        # Do Spatial Normalization (if needed)
        # ====================================
        norm_out = rt_snorm_vol(np.squeeze(smooth_out), do_operation=self.do_snorm)
        # Just putting an extra value on the end of each list in Data_norm. Not sure what this does
        # norm_out is not used elsewhere in preproc, nor is it indexed into in Data_norm
        self.Data_norm = np.append(self.Data_norm, norm_out, axis=1)

        if self.exp_type == "esam" or self.exp_type == "esam_test":

            # Do Windowing (if needed)
            # ========================
            if (self.t >= self.dec_start_vol) and self.hit_dowin:
                vols_in_win       = (np.arange(self.t-4,self.t)+1)[::-1]
                out_data_windowed = np.dot(self.Data_norm[:,vols_in_win], self.hit_win_weights)
                #self.Data_wind    = np.append(self.Data_wind, out_data_windowed, axis = 1)
            else:
                out_data_windowed = norm_out
                #self.Data_wind    = np.append(self.Data_wind, norm_out, axis =1)
            log.debug('[t=%d,n=%d] Online - SVRs - out_data_windowed.shape   %s' % (self.t, self.n, str(out_data_windowed.shape)))

            # Compute SVR scores (if needed)
            # ==============================
            if self.t < self.dec_start_vol:   # We don't want to start decoding before iGLM is stable.
                self.svrscores = np.append(self.svrscores, np.zeros((self.Ncaps,1)), axis=1)
            else:
                this_t_svrscores = rt_svrscore_vol(np.squeeze(out_data_windowed), self.SVRs, self.caps_labels)
                self.svrscores   = np.append(self.svrscores, this_t_svrscores, axis=1)
            log.debug('[t=%d,n=%d] Online - SVRs - svrscores.shape   %s' % (self.t, self.n, str(self.svrscores.shape)))

            # Compute Hits (if needed)
            # ========================
            # IF QA ended during the past TR (as I saved the status as soon as compute_TR started), then
            # update the last_QA_endTR and clear events
            if qa_end_status is True:
                self.lastQA_endTR = self.t
                self.qa_offsets.append(self.t)
                self.mp_evt_qa_end.clear()
                log.info(' - compute_TR_data - QA ended (cleared) --> updating lastQA_endTR = %d' % self.lastQA_endTR)

            # IF needed (e.g., not in hit mode, and late enough since last time), then compute if
            # a hit is taking place or not.
            if (hit_status == True) or (self.t <= self.lastQA_endTR + self.vols_noqa):
                hit = None
            else:
                hit = self.hit_method_func(
                    self.t,
                    self.caps_labels,
                    self.svrscores,
                    self.hit_zth,
                    self.nconsec_vols
                )
            
            # Add one more line to the hits data structure with zeros (if a hit happen, a 1 will be added later)
            self.hits = np.append(self.hits, np.zeros((self.Ncaps,1)), axis=1)

            # If there was a hit, then add that one, inform the use, and set the hit event to true
            if hit is not None:
                #if (hit != None) and ( hit_status == False ) and (self.t >= self.lastQA_endTR + self.vols_noqa):
                log.info('[t=%d,n=%d] =============================================  CAP hit [%s]' % (self.t,self.n, hit))
                self.qa_onsets.append(self.t)
                self.hits[self.caps_labels.index(hit),self.t] = 1
                self.mp_evt_hit.set()

        return 1

    def final_steps(self):
        # Write out motion
        self.motion_estimates = [item for sublist in self.motion_estimates for item in sublist]
        log.info('self.motion_estimates length is %d' % len(self.motion_estimates))
        self.motion_estimates = np.reshape(self.motion_estimates,newshape=(int(len(self.motion_estimates)/6),6))
        np.savetxt(osp.join(self.out_dir,self.out_prefix+'.Motion.1D'), 
                   self.motion_estimates,
                   delimiter="\t")
        log.info('Motion estimates saved to disk: [%s]' % osp.join(self.out_dir,self.out_prefix+'.Motion.1D'))

        if self.mask_img is None:
            log.warning(' final_steps = No additional outputs generated due to lack of mask.')
            return 1
        
        log.debug(' final_steps - About to write outputs to disk.')
        out_vars   = [self.Data_norm]
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

        if self.exp_type == "esam" or self.exp_type == "esam_test":
            svrscores_path = osp.join(self.out_dir,self.out_prefix+'.svrscores')
            np.save(svrscores_path,self.svrscores)
            log.info('Saved svrscores to %s' % svrscores_path)
            hits_path = osp.join(self.out_dir,self.out_prefix+'.hits')
            np.save(hits_path, self.hits)
            log.info('Saved hits info to %s' % hits_path)
        log.info(' - final_steps - Setting end of experiment event (mp_evt_end)')

        # Write out the dynamic report for this run
        if self.exp_type == "esam" or self.exp_type == "esam_test":
            SVR_Scores_DF       = pd.DataFrame(self.svrscores.T, columns=CAP_labels)
            SVR_Scores_DF['TR'] = SVR_Scores_DF.index
            SVRscores_curve     = SVR_Scores_DF.hvplot(legend='top', label='SVR Scores', x='TR').opts(width=1500)
            Threshold_line      = hv.HLine(self.hit_zth).opts(color='black', line_dash='dashed', line_width=1)
            Hits_ToPlot         = self.hits.T * self.svrscores.T
            Hits_ToPlot[Hits_ToPlot==0.0] = None
            Hits_DF             = pd.DataFrame(Hits_ToPlot, columns=CAP_labels)
            Hits_DF['TR']       = Hits_DF.index
            Hits_Marks          = Hits_DF.hvplot(legend='top', label='SVR Scores', 
                                             x='TR', kind='scatter', marker='circle', 
                                             alpha=0.5, s=100).opts(width=1500)
            qa_boxes = []
            for (on,off) in zip(self.qa_onsets,self.qa_offsets):
                qa_boxes.append(hv.Box(x=on+((off-on)/2),y=0,spec=(off-on,10)))
            QA_periods = hv.Polygons(qa_boxes).opts(alpha=.2, color='blue', line_color=None)
            wait_boxes = []
            for off in self.qa_offsets:
                wait_boxes.append(hv.Box(x=off+(self.vols_noqa/2),y=0,spec=(self.vols_noqa,10)))
            WAIT_periods = hv.Polygons(wait_boxes).opts(alpha=.2, color='cyan', line_color=None)
            plot_layout = (SVRscores_curve * Threshold_line * Hits_Marks * QA_periods * WAIT_periods).opts(title='Experimental Run Results:'+self.out_prefix)
            renderer    = hv.renderer('bokeh')
            renderer.save(plot_layout, self.outhtml)
            log.info(' - final_steps - Dynamic Report written to disk: [%s.html]' % self.outhtml)
            log.info(' - final_steps - qa_onsets:  %s' % str(self.qa_onsets))
            log.info(' - final_steps - qa_offsets: %s' % str(self.qa_offsets))

        # Write out the maps associated with the hits
        if self.exp_type == "esam" or self.exp_type == "esam_test":
            Hits_DF = pd.DataFrame(self.hits.T, columns=CAP_labels)
            for cap in CAP_labels:
                thisCAP_hits = Hits_DF[cap].sum()
                if thisCAP_hits > 0: # There were hits for this particular cap
                    hit_ID = 1
                    for vol in Hits_DF[Hits_DF[cap]==True].index:
                        if self.hit_dowin == True:
                            thisCAP_Vols = vol-np.arange(self.hit_wl+self.nconsec_vols-1)
                        else:
                            thisCAP_Vols = vol-np.arange(self.nconsec_vols)
                        out_file = osp.join(self.out_dir, self.out_prefix + '.Hit_'+cap+'_'+str(hit_ID).zfill(2)+'.nii')
                        log.info(' - final_steps - [%s-%d]. Contributing Vols: %s | File: %s' % (cap, hit_ID,str(thisCAP_Vols), out_file ))
                        log.debug(' - final_steps - self.Data_norm.shape %s' % str(self.Data_norm.shape))
                        log.debug(' - final_steps - self.Data_norm[:,thisCAP_Vols].shape %s' % str(self.Data_norm[:,thisCAP_Vols].shape))
                        thisCAP_InMask  = self.Data_norm[:,thisCAP_Vols].mean(axis=1)
                        log.debug(' - final_steps - thisCAP_InMask.shape %s' % str(thisCAP_InMask.shape))
                        unmask_fMRI_img(thisCAP_InMask, self.mask_img, out_file)
                        hit_ID = hit_ID + 1

            # Write out QA_Onsers and QA_Offsets
            with open(self.qa_onsets_path,'w') as file:
                for onset in self.qa_onsets:
                    file.write("%i\n" % onset)
            with open(self.qa_offsets_path,'w') as file:
                for offset in self.qa_offsets:
                    file.write("%i\n" % offset)        
        
        # If running snapshot test, save the variable states
        if self.snapshot:
            var_dict = {
                'Data_norm': self.Data_norm,
                'Data_EMA': self.Data_EMA,
                'Data_iGLM': self.Data_iGLM,
                'Data_smooth': self.Data_smooth,
                # 'Data_kalman': self.Data_kalman,
                'Data_FromAFNI': self.Data_FromAFNI
            }

            np.savez(osp.join(self.out_dir, f'{self.out_prefix}_snapshots.npz'), **var_dict) 

        # Inform other threads that this is comming to an end
        self.mp_evt_end.set()
        return 1

    def setup_esam_run(self, options):
        # load SVR model
        if options.svr_path is None:
            log.error('SVR Model not provided. Program will exit.')
            self.mp_evt_end.set()
            sys.exit(-1)
        if not osp.exists(options.svr_path):
            log.error('SVR Model File does not exists. Please correct.')
            self.mp_evt_end.set()
            sys.exit(-1)
        self.svr_path = options.svr_path
        try:
            SVRs_pickle_in = open(self.svr_path, "rb")
            self.SVRs = pickle.load(SVRs_pickle_in)
        except OSError as ose:
            log.error('SVR Model File opening threw OSError Exception.')
            log.error(traceback.format_exc(ose))
            self.mp_evt_end.set()
            sys.exit(-1)
        except Exception as e:
            log.error('SVR Model File opening threw generic Exception.')
            log.error(traceback.format_exc(e))
            self.mp_evt_end.set()
            sys.exit(-1)

        self.Ncaps = len(self.SVRs.keys())
        self.caps_labels = list(self.SVRs.keys())
        log.info('- setup_esam_run - List of CAPs to be tested: %s' % str(self.caps_labels))

        # Decoder-related initializations
        self.dec_start_vol = options.dec_start_vol # First volume to do decoding on.
        self.hit_method    = options.hit_method
        self.hit_zth       = options.hit_zth
        self.nconsec_vols  = options.nconsec_vols
        self.hit_dowin     = options.hit_dowin
        self.hit_domot     = options.hit_domot
        self.hit_mot_th    = options.svr_mot_th
        self.hit_wl        = options.hit_wl
        if self.hit_dowin:
            self.hit_win_weights = create_win(self.hit_wl)
        self.hit_method_func = None
        if self.hit_method == "method01":
            self.hit_method_func = is_hit_rt01

        return 1

# ==================================================
# ======== Functions (for Comm Process)   ==========
# ==================================================
def comm_process(opts, mp_evt_hit, mp_evt_end, mp_evt_qa_end):
    
    # 2) Create Experiment Object
    log.info('- comm_process - 2) Instantiating Experiment Object...')
    experiment = Experiment(opts, mp_evt_hit, mp_evt_end, mp_evt_qa_end)
 
    # 3) Initilize GUI (if needed):
    if experiment.exp_type == "esam" or experiment.exp_type == "esam_test":
        log.info('- comm_process - 2.a) This is an experimental run')
        experiment.setup_esam_run(opts) 
    
    # 4) Start Communications
    log.info('- comm_process - 3) Opening Communication Channel...')
    receiver = CustomReceiverInterface(port=opts.tcp_port, show_data=opts.show_data)
    # receiver = ReceiverInterface(port=opts.tcp_port, show_data=opts.show_data)
    if not receiver:
        return 1

    if not receiver.RTI:
        log.error('comm_process - RTI is not initialized.')
    else:
        log.debug('comm_process - RTI initialized successfully.')

    if not receiver:
        return 1


    # 5) set signal handlers and look for data
    log.info('- comm_process - 4) Setting Signal Handlers...')
    receiver.set_signal_handlers()

    # 6) set receiver callback
    receiver.compute_TR_data  = experiment.compute_TR_data
    receiver.final_steps      = experiment.final_steps

    # 7) prepare for incoming connections
    log.info('- comm_process - 5) Prepare for Incoming Connections...')
    if receiver.RTI.open_incoming_socket():
        return 1
    
    #8) Vinai's alternative
    log.info('6) Here we go...')
    rv = receiver.process_one_run()

    if experiment.exp_type == "esam" or experiment.exp_type == "esam_test":
        while experiment.mp_evt_hit.is_set():
            log.info('- comm_process - waiting for QA to end ')
            sleep(1)
    log.info('- comm_process - ready to end ')
    return rv


def processExperimentOptions (self, options=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("-c", "--config", dest="config", help="JSON file containing experiment options")
    pre_args, _ = pre_parser.parse_known_args(options)

    if pre_args.config:
        # Load options from json file and return as Namespace
        with open(pre_args.config, 'r') as f:
            config_data = json.load(f)
        return argparse.Namespace(**config_data)

    # If no json file is provided, parse args as usual
    parser = argparse.ArgumentParser(
    description="rtCAPs experimental software. Based on NIH-neurofeedback software. "
                "You can optionally provide a JSON config file via --config to set all options."
    )

    parser_gen = parser.add_argument_group("General Options")
    parser_gen.add_argument("-d", "--debug", action="store_true", dest="debug",  help="Enable debugging output [%(default)s]", default=False)
    parser_gen.add_argument("-s", "--silent",   action="store_true", dest="silent", help="Minimal text messages [%(default)s]", default=False)
    parser_gen.add_argument("-p", "--tcp_port", help="TCP port for incoming connections [%(default)s]", action="store", default=53214, type=int, dest='tcp_port')
    parser_gen.add_argument("-S", "--show_data", action="store_true",help="display received data in terminal if this option is specified")
    parser_gen.add_argument("--tr",         help="Repetition time [sec]  [default: %(default)s]",                      dest="tr",default=1.0, action="store", type=float)
    parser_gen.add_argument("--ncores",     help="Number of cores to use in the parallel processing part of the code  [default: %(default)s]", dest="n_cores", action="store",type=int, default=10)
    parser_gen.add_argument("--mask",       help="Mask necessary for smoothing operation  [default: %(default)s]",     dest="mask_path", action="store", type=str, default=None, required=True)
    parser_proc   = parser.add_argument_group("Activate/Deactivate Processing Steps")
    parser_proc.add_argument("--no_ema",    help="De-activate EMA Filtering Step [default: %(default)s]", dest="do_EMA",      default=True, action="store_false")
    parser_proc.add_argument("--no_iglm",   help="De-activate iGLM Denoising Step  [default: %(default)s]",             dest="do_iGLM",     default=True, action="store_false")
    parser_proc.add_argument("--no_kalman", help="De-activate Kalman Low-Pass Filter Step  [default: %(default)s]",     dest="do_kalman",   default=True, action="store_false")
    parser_proc.add_argument("--no_smooth", help="De-activate Spatial Smoothing Step  [default: %(default)s]",          dest="do_smooth",   default=True, action="store_false")
    parser_proc.add_argument("--no_snorm",  help="De-activate per-volume spartial Z-Scoring  [default: %(default)s]",   dest="do_snorm",   default=True, action="store_false")
    parser_iglm = parser.add_argument_group("Incremental GLM Options")
    parser_iglm.add_argument("--polort",     help="Order of Legengre Polynomials for iGLM  [default: %(default)s]",     dest="iGLM_polort", default=2, action="store", type=int)
    parser_iglm.add_argument("--no_iglm_motion", help="Do not use 6 motion parameters in iGLM  [default: %(default)s]", dest="iGLM_motion", default=True, action="store_false")
    parser_iglm.add_argument("--nvols",      help="Number of expected volumes (for legendre pols only)  [default: %(default)s]", dest="nvols",default=500, action="store", type=int, required=True)
    parser_iglm.add_argument("--discard",    help="Number of volumes to discard (they won't enter the iGLM step)  [default: %(default)s]",  default=10, dest="discard", action="store", type=int)
    parser_smo = parser.add_argument_group("Smoothing Options")
    parser_smo.add_argument("--fwhm",      help="FWHM for Spatial Smoothing in [mm]  [default: %(default)s]",          dest="FWHM",        default=4.0, action="store", type=float)
    parser_save   = parser.add_argument_group("Saving Options")
    parser_save.add_argument("--out_dir",     help="Output directory  [default: %(default)s]",                           dest="out_dir",    action="store", type=str, default="./")
    parser_save.add_argument("--out_prefix",  help="Prefix for outputs  [default: %(default)s]",                         dest="out_prefix", action="store", type=str, default="online_preproc")
    parser_save.add_argument("--save_ema",    help="Save 4D EMA dataset  [default: %(default)s]",     dest="save_ema",   default=False, action="store_true")
    parser_save.add_argument("--save_kalman", help="Save 4D Smooth dataset  [default: %(default)s]",     dest="save_kalman",   default=False, action="store_true")
    parser_save.add_argument("--save_smooth", help="Save 4D Smooth dataset  [default: %(default)s]",     dest="save_smooth",   default=False, action="store_true")
    parser_save.add_argument("--save_iglm  ", help="Save 4D iGLM datasets  [default: %(default)s]",     dest="save_iglm",   default=False, action="store_true")
    parser_save.add_argument("--save_orig"  , help="Save 4D with incoming data  [default: %(default)s]", dest="save_orig", default=False, action="store_true")
    parser_save.add_argument("--save_all"  ,  help="Save 4D with incoming data  [default: %(default)s]", dest="save_all", default=False, action="store_true")
    parser_exp = parser.add_argument_group('Experiment/GUI Options')
    parser_exp.add_argument("-e","--exp_type", help="Type of Experimental Run [%(default)s]",      type=str, required=True,  choices=['preproc','esam', 'esam_test'], default='preproc')
    parser_exp.add_argument("--no_gui", help="Do not open psychopy window. Only applies to pre-processing experiment type [%(default)s]", default=False, action="store_true", dest='no_gui')
    parser_exp.add_argument("--no_proc_chair", help="Hide crosshair during preprocessing run [%(default)s]", default=False,  action="store_true", dest='no_proc_chair')
    parser_exp.add_argument("--fscreen", help="Use full screen for Experiment [%(default)s]", default=False, action="store_true", dest="fullscreen")
    parser_exp.add_argument("--screen", help="Monitor to use [%(default)s]", default=1, action="store", dest="screen",type=int)
    parser_dec = parser.add_argument_group('SVR/Decoding Options')
    parser_dec.add_argument("--svr_start",  help="Volume when decoding should start. When we think iGLM is sufficient_stable [%(default)s]", default=100, dest="dec_start_vol", action="store", type=int)
    parser_dec.add_argument("--svr_path",   help="Path to pre-trained SVR models [%(default)s]", dest="svr_path", action="store", type=str, default=None)
    parser_dec.add_argument("--svr_zth",    help="Z-score threshold for deciding hits [%(default)s]", dest="hit_zth", action="store", type=float, default=1.75)
    parser_dec.add_argument("--svr_consec_vols",   help="Number of consecutive vols over threshold required for a hit [%(default)s]", dest="nconsec_vols", action="store", type=int, default=2)
    parser_dec.add_argument("--svr_win_activate", help="Activate windowing of individual volumes prior to hit estimation [%(default)s]", dest="hit_dowin", action="store_true", default=False)
    parser_dec.add_argument("--svr_win_wl", help='Number of volumes for SVR windowing step [%(default)s]', dest='hit_wl', default=4, type=int, action='store')
    parser_dec.add_argument("--svr_mot_activate", help="Consider a hit if excessive motion [%(default)s]", dest="hit_domot", action="store_true", default=False )
    parser_dec.add_argument("--svr_mot_th", help="Framewise Displacement Treshold for motion [%(default)s]",  action="store", type=float, dest="svr_mot_th", default=1.2)
    parser_dec.add_argument("--svr_hit_mehod", help="Method for deciding hits [%(default)s]", type=str, choices=["method01"], default="method01", action="store", dest="hit_method")
    parser_dec.add_argument("--svr_vols_noqa", help="Min. number of volumes to wait since end of last QA before declaing a new hit. [%(default)s]", type=int, dest='vols_noqa', default=45, action="store"),
    parser_dec.add_argument("--q_path", help="The path to the questions json file containing the question stimuli. If not a full path, it will default to look in RESOURCES_DIR", type=str, dest='q_path', default="questions_v1", action="store")
    parser_dec = parser.add_argument_group('Testing Options')
    parser_dec.add_argument("--snapshot",  help="Run snapshot test", default=False, dest="snapshot", action="store_true")

    return parser.parse_args(options)

def main():
    # 1) Read Input Parameters: port, fullscreen, etc..
    # -------------------------------------------------
    log.info('1) Reading input parameters...')
    opts = processExperimentOptions(sys.argv)
    log.debug('User Options: %s' % str(opts))

    # Load Likert questions before starting up GUI
    if opts.exp_type in ["esam", "esam_test"]:
        if not opts.q_path:
            log.error('Path to Likert questions was not provided. Program will exit.')
            sys.exit(-1)
        if not osp.isfile(opts.q_path): # If not file, assume in RESOURCES_DIR
            fname = opts.q_path + ".json" if not opts.q_path.endswith(".json") else opts.q_path 
            opts.q_path = osp.join(RESOURCES_DIR, fname)
        try:
            with open(opts.q_path, 'r') as f:
                opts.likert_questions = json.load(f)
        except json.JSONDecodeError:
            log.error(f'The question file at {opts.q_path} is not a valid JSON.')
            sys.exit(-1)
        except Exception as e:
            log.error(f'Error loading questions at {opts.q_path}: {e}')
            sys.exit(-1)

    opts_tofile_path = osp.join(opts.out_dir, opts.out_prefix+'_Options.json')
    with open(opts_tofile_path, "w") as write_file:
        json.dump(vars(opts), write_file)
    log.info('  - Options written to disk [%s]'% opts_tofile_path)
    
    # 3) Create Multi-processing infrastructure
    # ------------------------------------------
    mp_evt_hit    = mp.Event() # Start with false
    mp_evt_end    = mp.Event() # Start with false
    mp_evt_qa_end = mp.Event() # Start with false
    mp_prc_comm   = mp.Process(target=comm_process, args=(opts, mp_evt_hit, mp_evt_end, mp_evt_qa_end))
    mp_prc_comm.start()

    # 2) Get additional info using the GUI
    # ------------------------------------
    if not opts.no_gui:
        exp_info = get_experiment_info(opts)

    # 3) Depending on the type of run.....
    # ------------------------------------
    if opts.exp_type == "esam":
        # 4) Start GUI
        # ------------
        cap_qa = QAScreen(exp_info, opts)
    
        # 5) Wait for things to happen
        # ----------------------------
        while not mp_evt_end.is_set():
            cap_qa.draw_resting_screen()
            if event.getKeys(['escape']):
                log.info('- User pressed escape key')
                mp_evt_end.set()
            if mp_evt_hit.is_set():
                responses = cap_qa.run_full_QA()
                log.info(' - Responses: %s' % str(responses))
                mp_evt_hit.clear()
                mp_evt_qa_end.set()
        
        # 6) Close Psychopy Window
        # ------------------------
        cap_qa.save_likert_files()
        cap_qa.close_psychopy_infrastructure()
        
    if opts.exp_type == "esam_test":
        # 4) Start GUI
        # ------------
        cap_qa = QAScreen(exp_info,opts)
    
        # 5) Wait for things to happen
        # ----------------------------
        while not mp_evt_end.is_set():
            cap_qa.draw_resting_screen()
            if event.getKeys(['escape']):
                log.info('- User pressed escape key')
                mp_evt_end.set()
       
        # 6) Close Psychopy Window
        # ------------------------
        cap_qa.close_psychopy_infrastructure()
    
    if opts.exp_type == "preproc":
        if not opts.no_gui:
            # 4) Start GUI
            rest_exp = DefaultScreen(exp_info, opts)

            # 5) Keep the experiment going, until it ends
            while not mp_evt_end.is_set():
                rest_exp.draw_resting_screen()
                if event.getKeys(['escape']):
                    log.info('- User pressed escape key')
                    mp_evt_end.set()

            # 6) Close Psychopy Window
            # ------------------------
            rest_exp.close_psychopy_infrastructure()
        else:
            # 4) In no_gui mode, wait passively for experiment to end
            while not mp_evt_end.is_set():
                time.sleep(0.1)
        
        
    log.info(' - main - Reached end of Main in primary thread')
    return 1

if __name__ == '__main__':
    sys.exit(main())