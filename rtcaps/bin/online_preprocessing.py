#!/usr/bin/env python

# python3 status: compatible

import sys, os
import itertools
import signal, time
import argparse
import logging
import pickle
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import numpy as np
import multiprocessing
#multiprocessing.set_start_method('spawn', True)
import os.path as osp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from afni_lib.receiver import ReceiverInterface
from rtcap_lib.core import unpack_extra, welford
from rtcap_lib.rt_functions import rt_EMA_vol, rt_regress_vol, rt_kalman_vol
from rtcap_lib.rt_functions import rt_smooth_vol, rt_snorm_vol, rt_svrscore_vol
from rtcap_lib.rt_functions import gen_polort_regressors
from rtcap_lib.fMRI         import load_fMRI_file, unmask_fMRI_img
from rtcap_lib.svr_methods  import is_hit_rt01
from rtcap_lib.core         import create_win

from psychopy.visual import Window, TextStim

# ----------------------------------------------------------------------
# Default Login Options:
log     = logging.getLogger("online_preproc")
log_fmt = logging.Formatter('[%(levelname)s - Main]: %(message)s')
log_ch  = logging.StreamHandler()
log_ch.setFormatter(log_fmt)
log.setLevel(logging.INFO)
log.addHandler(log_ch)

# ----------------------------------------------------------------------
# In this module, handing signals and options.  Try to keep other
# operations in separate libraries (e.g. lib_realtime.py).
# ----------------------------------------------------------------------

class Experiment(object):

    def __init__(self, options):

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

        # Load Mask - Necessary for smoothing
        if self.do_smooth and self.mask_path is None:
            log.error('   Experiment_init_ - Smoothing requires a mask. Provide a mask or disable smoothing operation.')
            sys.exit(-1)
        
        if self.mask_path is None:
            self.mask_img = None
        else:
            self.mask_img  = load_fMRI_file(self.mask_path)
            self.mask_Nv = np.sum(self.mask_img.get_data())
            log.debug('  Experiment_init_ - Number of Voxels in user-provided mask: %d' % self.mask_Nv)

        

        # Create Legendre Polynomial regressors
        if self.iGLM_polort > -1:
            self.legendre_pols = gen_polort_regressors(self.iGLM_polort,self.Nt)
        else:
            self.legendre_pols = None

        # If kalman needed, create a pool
        if self.do_kalman:
            self.pool = multiprocessing.Pool(processes=self.n_cores)
        else:
            self.pool = None

    def setup_preproc_withscreen_run(self):
        #create a window
        if self.ewin is None:
            if self.fullscreen:
                self.ewin = Window(fullscr = self.fullscreen, allowGUI=False, units='norm')
            else:
                self.ewin = Window(self.screen_size, allowGUI=False, units='norm')
        
        #create some stimuli
        text_inst_line01 = TextStim(win=self.ewin, text='Please fixate on x-hair,',pos=(0.0,0.4))
        text_inst_line02 = TextStim(win=self.ewin, text='remain awake,',           pos=(0.0,0.28))
        text_inst_line03 = TextStim(win=self.ewin, text='and let your mind wander.',pos=(0.0,0.16))
        text_inst_chair  = TextStim(win=self.ewin, text='X', pos=(0,0))

        #plot on the screen
        text_inst_line01.draw()
        text_inst_line02.draw()
        text_inst_line03.draw()
        text_inst_chair.draw()
        self.ewin.flip()
    
    def setup_esam_run(self, options):
        # load SVR model
        if options.svr_path is None:
            log.error('SVR Model not provided. Program will exit.')
            sys.exit(-1)
        if not osp.exists(options.svr_path):
            log.error('SVR Model File does not exists. Please correct.')
            sys.exit(-1)
        self.svr_path = options.svr_path
        try:
            SVRs_pickle_in = open(self.svr_path, "rb")
            self.SVRs = pickle.load(SVRs_pickle_in)
        except OSError as ose:
            log.error('SVR Model File opening threw OSError Exception.')
            log.error(traceback.format_exc(ose))
            sys.exit(-1)
        except Exception as e:
            log.error('SVR Model File opening threw generic Exception.')
            log.error(traceback.format_exc(e))
            sys.exit(-1)
        self.Ncaps = len(self.SVRs.keys())
        self.caps_labels = list(self.SVRs.keys())
        log.info('- setup_esam_run - List of CAPs to be tested: %s' % str(self.caps_labels))
        
        # Decoder-related initializations
        self.dec_start_vol = options.dec_start_vol # First volume to do decoding on.
        self.hit_method    = options.hit_method
        self.hit_zth       = options.hit_zth
        self.hit_v4hit     = options.hit_v4hit
        self.hit_dowin     = options.hit_dowin
        self.hit_domot     = options.hit_domot
        self.hit_mot_th    = options.svr_mot_th
        self.hit_wl        = options.hit_wl
        if self.hit_dowin:
            self.hit_win_weights = create_win(self.hit_wl)
        self.hit_method_func = None
        if self.hit_method == "method01":
            self.hit_method_func = is_hit_rt01

        # create initial window with instructions
        self.setup_preproc_withscreen_run()

    def compute_TR_data(self, motion, extra):
        # Keep a record of motion estimates
        self.motion_estimates.append(motion)
        # Update t (it always does)
        self.t = self.t + 1

        # Update n (only if not longer a discard volume)
        if self.t > self.nvols_discard - 1:
            self.n = self.n + 1

        log.info(' - Time point [t=%d, n=%d]' % (self.t, self.n))
        
        # Read Data from socket
        this_t_data = np.array(extra)
        
        # If first volume, then create empty structures and call it a day (TR)
        if self.t == 0:
            self.Nv            = len(this_t_data)
            log.info('Number of Voxels Nv=%d' % self.Nv)
            if self.exp_type == "esam":
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
            self.S_x           = np.zeros(self.Nv) #[0]*self.Nv 
            self.S_P           = np.zeros(self.Nv) #[0]*self.Nv 
            self.fPositDerivSpike = np.zeros(self.Nv) #[0]*self.Nv 
            self.fNegatDerivSpike = np.zeros(self.Nv) #[0]*self.Nv
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
            if self.save_iGLM: self.iGLM_Coeffs   = np.append(self.iGLM_Coeffs, np.zeros((self.Nv,self.iGLM_num_regressors,1)), axis=2)
            log.debug('[t=%d,n=%d] Discard - Data_FromAFNI.shape %s' % (self.t, self.n, str(self.Data_FromAFNI.shape)))
            if self.save_ema:    log.debug('[t=%d,n=%d] Discard - Data_EMA.shape      %s' % (self.t, self.n, str(self.Data_EMA.shape)))
            if self.save_iGLM:   log.debug('[t=%d,n=%d] Discard - Data_iGLM.shape     %s' % (self.t, self.n, str(self.Data_iGLM.shape)))
            if self.save_kalman: log.debug('[t=%d,n=%d] Discard - Data_kalman.shape   %s' % (self.t, self.n, str(self.Data_kalman.shape)))
            log.debug('[t=%d,n=%d] Discard - Data_norm.shape     %s' % (self.t, self.n, str(self.Data_norm.shape)))
            if self.save_iGLM: log.debug('[t=%d,n=%d] Discard - iGLM_Coeffs.shape   %s' % (self.t, self.n, str(self.iGLM_Coeffs.shape)))
            if self.exp_type == "esam":
                # These two variables are only needed if this is an experimental
                self.hits      = np.append(self.hits,      np.zeros((self.Ncaps,1)),  axis=1)
                self.svrscores = np.append(self.svrscores, np.zeros((self.Ncaps,1)), axis=1)
                log.debug('[t=%d,n=%d] Discard - hits.shape      %s' % (self.t, self.n, str(self.hits.shape)))
                log.debug('[t=%d,n=%d] Discard - svrscores.shape %s' % (self.t, self.n, str(self.svrscores.shape)))
            return 1

        # Compute running mean and running std with welford
        self.welford_M, self.welford_S, self.weldord_std = welford(self.n, this_t_data, self.welford_M, self.welford_S)
        
        # If we reach this point, it means we have work to do
        if self.save_orig:
            self.Data_FromAFNI = np.append(self.Data_FromAFNI,this_t_data[:, np.newaxis], axis=1)
        else:
            self.Data_FromAFNI = np.hstack((self.Data_FromAFNI[:,-1][:,np.newaxis],this_t_data[:, np.newaxis]))  # Only keep this one and previous
            log.debug('[t=%d,n=%d] Online - Input - Data_FromAFNI.shape %s' % (self.t, self.n, str(self.Data_FromAFNI.shape)))

        # Do EMA (if needed)
        # ==================
        ema_data_out, self.EMA_filt = rt_EMA_vol(self.n, self.t, self.EMA_th, self.Data_FromAFNI, self.EMA_filt, do_operation = self.do_EMA)
        #log.debug('[t=%d,n=%d] Online - EMA - ema_data_out.shape      %s' % (self.t, self.n, str(ema_data_out.shape)))
        if self.save_ema: 
            self.Data_EMA = np.append(self.Data_EMA, ema_data_out, axis=1)
            log.debug('[t=%d,n=%d] Online - EMA - Data_EMA.shape      %s' % (self.t, self.n, str(self.Data_EMA.shape)))
        
        # Do iGLM (if needed)
        # ===================
        if self.iGLM_motion:
            this_t_nuisance = np.concatenate((self.legendre_pols[self.t,:],motion))[:,np.newaxis]
        else:
            this_t_nuisance = (self.legendre_pols[self.t,:])[:,np.newaxis]
        iGLM_data_out, self.iGLM_prev, Bn = rt_regress_vol(self.n, 
                                                           ema_data_out,
                                                           this_t_nuisance,
                                                           self.iGLM_prev,
                                                           do_operation = self.do_iGLM)
        if self.save_iGLM: 
            self.Data_iGLM    = np.append(self.Data_iGLM, iGLM_data_out, axis=1)
            self.iGLM_Coeffs  = np.append(self.iGLM_Coeffs, Bn, axis = 2) 
            log.debug('[t=%d,n=%d] Online - iGLM - Data_iGLM.shape     %s' % (self.t, self.n, str(self.Data_iGLM.shape)))
            log.debug('[t=%d,n=%d] Online - iGLM - iGLM_Coeffs.shape   %s' % (self.t, self.n, str(self.iGLM_Coeffs.shape)))

        # Do Kalman Low-Pass Filter (if needed)
        # =====================================
        klm_data_out, self.S_x, self.S_P, self.fPositDerivSpike, self.fNegatDerivSpike = rt_kalman_vol(self.n,
                                                                        self.t,
                                                                        iGLM_data_out,
                                                                        self.weldord_std,
                                                                        self.S_x,
                                                                        self.S_P,
                                                                        self.fPositDerivSpike,
                                                                        self.fNegatDerivSpike,
                                                                        self.n_cores,
                                                                        self.pool,
                                                                        do_operation = self.do_kalman)
                
        if self.save_kalman: 
            self.Data_kalman      = np.append(self.Data_kalman, klm_data_out, axis = 1)
            log.debug('[t=%d,n=%d] Online - Kalman - Data_kalman.shape     %s' % (self.t, self.n, str(self.Data_kalman.shape)))

        # Do Spatial Smoothing (if needed)
        # ================================
        smooth_out = rt_smooth_vol(np.squeeze(klm_data_out), self.mask_img, fwhm = self.FWHM, do_operation = self.do_smooth)
        if self.save_smooth:
            self.Data_smooth = np.append(self.Data_smooth, smooth_out, axis=1)
            log.debug('[t=%d,n=%d] Online - Smooth - Data_smooth.shape   %s' % (self.t, self.n, str(self.Data_smooth.shape)))
            log.debug('[t=%d,n=%d] Online - EMA - smooth_out.shape      %s' % (self.t, self.n, str(smooth_out.shape)))

        # Do Spatial Normalization (if needed)
        # ====================================
        norm_out = rt_snorm_vol(np.squeeze(smooth_out), do_operation=self.do_snorm)
        self.Data_norm = np.append(self.Data_norm, norm_out, axis=1)

        if self.exp_type == "esam":

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
                #this_t_svrscores = rt_svrscore_vol(self.Data_wind[:, self.t], self.SVRs, self.caps_labels)
                this_t_svrscores = rt_svrscore_vol(np.squeeze(out_data_windowed), self.SVRs, self.caps_labels)
                self.svrscores   = np.append(self.svrscores, this_t_svrscores, axis=1)
            log.debug('[t=%d,n=%d] Online - SVRs - svrscores.shape   %s' % (self.t, self.n, str(self.svrscores.shape)))

            # Compute Hits (if needed)
            # ========================
            # MISSING: Don't do this if before 100 vols
            hit = self.hit_method_func(self.t,
                                       self.caps_labels,
                                       self.svrscores,
                                       self.hit_zth,
                                       self.hit_wl)
            self.hits = np.append(self.hits, np.zeros((self.Ncaps,1)), axis=1)
            if hit != None:
                log.info('[t=%d,n=%d] =============================================  CAP hit [%s]' % (self.t,self.n, hit))
                self.hits[self.caps_labels.index(hit),self.t] = 1

        # Need to return something, otherwise the program thinks the experiment ended
        return 1

    def final_steps(self):
        if self.ewin is not None:
                log.info('Psychopy UI closing.')
                self.ewin.close()

        # Write out motion
        self.motion_estimates = [item for sublist in self.motion_estimates for item in sublist]
        log.info('self.motion_estimates length is %d' % len(self.motion_estimates))
        self.motion_estimates = np.reshape(self.motion_estimates,newshape=(int(len(self.motion_estimates)/6),6))
        np.savetxt(osp.join(self.out_dir,self.out_prefix+'Motion.1D'), 
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
        for variable, file_suffix in zip(out_vars, out_labels):
            out = unmask_fMRI_img(variable, self.mask_img, osp.join(self.out_dir,self.out_prefix+file_suffix))

        if self.do_iGLM and self.save_iGLM:
            for i,lab in enumerate(self.nuisance_labels):
                data = self.iGLM_Coeffs[:,i,:]
                out = unmask_fMRI_img(data, self.mask_img, osp.join(self.out_dir,self.out_prefix+'.pp_iGLM_'+lab+'.nii'))    

        if self.exp_type == "esam":
            svrscores_path = osp.join(self.out_dir,self.out_prefix+'.svrscores')
            np.save(svrscores_path,self.svrscores)
            log.info('Saved svrscores to %s' % svrscores_path)
            hits_path = osp.join(self.out_dir,self.out_prefix+'.hits')
            np.save(hits_path, self.hits)
            log.info('Saved hits info to %s' % hits_path)

        return 1

def processExperimentOptions (self, options=None):
    parser = argparse.ArgumentParser(description="rtCAPs experimental software. Based on NIH-neurofeedback software")
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
    parser_exp.add_argument("-e","--exp_type", help="Type of Experimental Run [%(default)s]",      type=str, required=True,  choices=['preproc','esam'], default='preproc')
    parser_exp.add_argument("--no_proc_chair", help="Hide crosshair during preprocessing run [%(default)s]", default=False,  action="store_true", dest='no_proc_chair')
    parser_exp.add_argument("--fscreen", help="Use full screen for Experiment [%(default)s]", default=False, action="store_true", dest="fullscreen")
    parser_exp.add_argument("--screen", help="Monitor to use [%(default)s]", default=1, action="store", dest="screen",type=int)
    parser_dec = parser.add_argument_group('SVR/Decoding Options')
    parser_dec.add_argument("--svr_start",  help="Volume when decoding should start. When we think iGLM is sufficient_stable [%(default)s]", default=100, dest="dec_start_vol", action="store", type=int)
    parser_dec.add_argument("--svr_path",   help="Path to pre-trained SVR models [%(default)s]", dest="svr_path", action="store", type=str, default=None)
    parser_dec.add_argument("--svr_zth",    help="Z-score threshold for deciding hits [%(default)s]", dest="hit_zth", action="store", type=float, default=2.0)
    parser_dec.add_argument("--svr_vhit",   help="Number of consecutive vols over threshold required for a hit [%(default)s]", dest="hit_v4hit", action="store", type=int, default=2)
    parser_dec.add_argument("--svr_win_activate", help="Activate windowing of individual volumes prior to hit estimation [%(default)s]", dest="hit_dowin", action="store_true", default=False)
    parser_dec.add_argument("--svr_win_wl", help='Number of volumes for SVR windowing step [%(default)s]', dest='hit_wl', default=4, type=int, action='store')
    parser_dec.add_argument("--svr_mot_activate", help="Consider a hit if excessive motion [%(default)s]", dest="hit_domot", action="store_true", default=False )
    parser_dec.add_argument("--svr_mot_th", help="Framewise Displacement Treshold for motion [%(default)s]",  action="store", type=float, dest="svr_mot_th", default=1.2)
    parser_dec.add_argument("--svr_hit_mehod", help="Method for deciding hits [%(default)s]", type=str, choices=["method01"], default="method01", action="store", dest="hit_method")
    #self.hit_method    = "method01"
    #    self.hit_zth       = 2
    #    self.hit_v4hit     = 2
    #    self.hit_dowin     = True
    #    self.hit_domot     = False
    #    self.hit_wl        = 4


    return parser.parse_args(options)

def main():
    # 1) Read Input Parameters: port, fullscreen, etc..
    log.info('1) Reading input parameters...')
    opts = processExperimentOptions(sys.argv)
    log.debug('User Options: %s' % str(opts))    

    # 2) Create Experiment Object
    log.info('2) Instantiating Experiment Object...')
    experiment = Experiment(opts)

    # 3) Initilize GUI (if needed):
    if (experiment.exp_type == "preproc") and (experiment.no_proc_chair==False):
        log.info('Starting Pychopy Screen for Experiment Run [ Preprocessing + Crosshiar ]')
        experiment.setup_preproc_withscreen_run()
    if experiment.exp_type == "esam":
        log.info('This is an experimental run')
        log.info('  - PsychoPy Screen Activated.')
        experiment.setup_esam_run(opts) 
        log.info('  - SVR Models loaded from %s' % experiment.svr_path)


    # 4) Start Communications
    log.info('3) Opening Communication Channel...')
    receiver = ReceiverInterface(port=opts.tcp_port, show_data=opts.show_data)
    if not receiver:
        return 1

    # 5) set signal handlers and look for data
    log.info('4) Setting Signal Handlers...')
    receiver.set_signal_handlers()  # require signal to exit

    # 6) set receiver callback
    # At this point Receiver is still basically an empty container
    receiver.compute_TR_data  = experiment.compute_TR_data
    receiver.final_steps      = experiment.final_steps

    # 7) prepare for incoming connections
    log.info('5) Prepare for Incoming Connections...')
    if receiver.RTI.open_incoming_socket():
        return 1

    #8) Vinai's alternative
    log.info('6) Here we go...')
    rv = receiver.process_one_run()
    return rv

if __name__ == '__main__':
   sys.exit(main())
