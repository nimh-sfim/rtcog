#!/usr/bin/env python

# python3 status: compatible
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import sys
import os
import signal
import time
import pickle
from optparse import OptionParser

import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', True)
import os.path as osp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                '..')))

from afni_lib.receiver import ReceiverInterface
# from afni_lib.receiver import ReceiverInterface
from rtcap_lib.core import unpack_extra
from rtcap_lib.rt_functions import rt_EMA_vol, rt_regress_vol, rt_kalman_vol 
from rtcap_lib.rt_functions import rt_smooth_vol, rt_snorm_vol, rt_svrscore_vol
from rtcap_lib.rt_functions import gen_polort_regressors
from rtcap_lib.fMRI import load_fMRI_file, unmask_fMRI_img
from rtcap_lib.core import create_win
from rtcap_lib.svr_methods  import is_hit_rt01

# Check if Psychopy is available (will move to a funcion later)
try:
    from psychopy import visual
    psychopyInstalled = 1
except ImportError:
    psychopyInstalled = 0

# ----------------------------------------------------------------------
# globals
# log = logging.getLogger(__name__)
# log.basicConfig(format='[%(levelname)s]: POP %(message)s', level=log.DEBUG)
log     = logging.getLogger("online_experiment")
log.setLevel(logging.INFO)
log_fmt = logging.Formatter('[%(levelname)s - Main]: %(message)s')
log_ch  = logging.StreamHandler()
log_ch.setFormatter(log_fmt)
log_ch.setLevel(logging.INFO)
log.addHandler(log_ch)

# ----------------------------------------------------------------------
# In this module, handing signals and options.  Try to keep other
# operations in separate libraries (e.g. lib_realtime.py).
# ----------------------------------------------------------------------

class Experiment(object):
    
    def __init__(self, options):
        self.n             = 0               # Counter for number of volumes pre-processed (Start = 1)
        self.t             = -1              # Counter for number of received volumes (Start = 0)
        self.Nv            = None            # Number of voxels in data mask
        self.Nt            = options.nvols   # Number acquisitions
        self.TR            = options.tr      # TR [seconds]
        self.n_cores       = options.n_cores # Number of cores for the multiprocessing part of the code
        self.Data_FromAFNI = None            # np.array [Nv,Nt] for incoming data
        self.save_ema      = opts.save_ema
        self.save_smooth   = opts.save_smooth

        if self.save_ema: self.Data_EMA      = None            # np.array [Nv,Nt] for data after EMA  step
        self.Data_iGLM     = None            # np.array [Nv,Nt] for data after iGLM step
        self.Data_kalma    = None            # np.array [Nv,Nt] for data after low-pass step
        if self.save_smooth: self.Data_smooth   = None            # np.array [Nv,Nt] for data after spatial smoothing
        self.Data_norm     = None            # np.array [Nv,Nt] for data after spatial normalization (spatial z-score)
        self.iGLM_Coeffs   = None            # np.array [Nregressor, Nv, Nt] for beta coefficients for all regressors
        
        self.do_EMA        = options.do_EMA       # Should we do EMA
        self.do_iGLM       = options.do_iGLM      # Should we do iGLM
        self.do_kalman     = options.do_kalman    # Should we do Low Pass Filter
        self.do_smooth     = options.do_smooth    # Should we do Spatial Filtering
        self.do_snorm      = options.do_snorm     # Should we do Spatial Z-scoring per volume
        self.FWHM          = options.FWHM         # FWHM for Spatial Smoothing in [mm]
        
        self.nvols_discard = options.discard   # Number of volumes to discard from any analysis (won't enter pre-processing)
        
        # Decoder-related initializations
        self.start_vol     = options.start_vol # First volume to do decoding on.
        self.hit_method    = "method01"
        self.hit_zth       = 2
        self.hit_v4hit     = 2
        self.hit_dowin     = True
        self.hit_domot     = False
        self.hit_wl        = 4
        if self.hit_dowin:
            self.hit_win_weights = create_win(self.hit_wl)
        self.hit_method_func = None
        if self.hit_method == "method01":
            self.hit_method_func = is_hit_rt01

        # iGLM-related initializations
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

        # Low-pass (kalman) Filter-related initializations
        self.S_x              = None
        self.S_P              = None
        self.fPositDerivSpike = None
        self.fNegatDerivSpike = None
        self.kalmThreshold    = None

        # EMA Filter-related initializations
        self.EMA_th   = 0.98
        self.EMA_filt = None

        # Smoothing & Output-related initalizations
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
            self.legendre_pols = gen_polort_regressors(self.iGLM_polort, self.Nt)
        else:
            self.legendre_pols = None

        # If kalman needed, create a pool
        if self.do_kalman:
            self.pool = multiprocessing.Pool(processes=self.n_cores)
        else:
            self.pool = None

        # if SVR models missing, then end program
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
        log.info('List of CAPs to be tested: %s' % str(self.caps_labels))

    def setup_experiment(self, options):
        self.fullscreen = options.fscreen
        self.screen_size = [512, 288]
        self.allowGUI   = False
        if self.fullscreen:
            self.ewin = visual.Window(fullscr=self.fullscreen, allowGUI=self.allowGUI)
        else:
            self.ewin = visual.Window(self.screen_size, allowGUI=self.allowGUI)
        self.gen_inst_msg = visual.TextStim(self.ewin,
                            text = "Please focus on the crosshair and let your mind wander freely.", 
                            bold = True,
                            pos = (0.0,0.5))
        self.crosshair = visual.TextStim(self.ewin,
                            text="+",
                            bold=True,
                            pos=(0.0,0.0),
                            height=.5)
        self.gen_inst_msg.draw()
        self.crosshair.draw()
        self.ewin.flip()
    

    def compute_TR_data(self, motion, extra):
        msg = visual.TextStim(self.ewin, text="%d" % self.t)
        msg.draw()
        self.ewin.flip()
        # Update t (it always does)
        self.t = self.t + 1
        # Update n (only if not longer a discard volume)
        if self.t > self.nvols_discard - 1:
            self.n = self.n + 1

        log.info(' - Time point [t=%d, n=%d]' % (self.t, self.n))
        
        # Read Data from socket
        this_t_data = unpack_extra(extra)
        
        # If first volume, then create empty structures and call it a day (TR)
        if self.t == 0:
            self.Nv            = len(this_t_data)
            if self.do_smooth:
                if self.mask_Nv != self.Nv:
                    log.error('Discrepancy across masks [data Nv = %d, mask Nv = %d]' % (self.Nv, self.mask_Nv) )
                    sys.exit(-1)
            self.Data_FromAFNI = np.array(this_t_data[:,np.newaxis])
            if self.save_ema: self.Data_EMA = np.zeros((self.Nv,1))
            self.Data_iGLM     = np.zeros((self.Nv,1))
            self.Data_kalman   = np.zeros((self.Nv,1))
            if self.save_smooth: self.Data_smooth   = np.zeros((self.Nv,1))
            self.Data_norm     = np.zeros((self.Nv,1))
            self.Data_wind     = np.zeros((self.Nv,1))
            self.nuisance      = np.zeros((self.iGLM_num_regressors,1))
            self.iGLM_Coeffs   = np.zeros((self.Nv,self.iGLM_num_regressors,1))
            self.S_x           = np.zeros((self.Nv,1))
            self.S_P           = np.zeros((self.Nv,1))
            self.fPositDerivSpike = np.zeros((self.Nv, 1))
            self.fNegatDerivSpike = np.zeros((self.Nv, 1))
            self.hits             = np.zeros((self.Ncaps, 1))
            self.svrscores        = np.zeros((self.Ncaps, 1))

            #self.kalmThreshold    = np.zeros((self.Nv,1))
            log.debug('[t=%d,n=%d] Init - Data_FromAFNI.shape %s' % (self.t, self.n, str(self.Data_FromAFNI.shape)))
            if self.save_ema: log.debug('[t=%d,n=%d] Init - Data_EMA.shape      %s' % (self.t, self.n, str(self.Data_EMA.shape)))
            log.debug('[t=%d,n=%d] Init - Data_iGLM.shape     %s' % (self.t, self.n, str(self.Data_iGLM.shape)))
            log.debug('[t=%d,n=%d] Init - Data_norm.shape     %s' % (self.t, self.n, str(self.Data_norm.shape)))
            log.debug('[t=%d,n=%d] Init - Data_wind.shape     %s' % (self.t, self.n, str(self.Data_wind.shape)))
            log.debug('[t=%d,n=%d] Init - nuisance.shape      %s' % (self.t, self.n, str(self.nuisance.shape)))
            log.debug('[t=%d,n=%d] Init - iGLM_Coeffs.shape   %s' % (self.t, self.n, str(self.iGLM_Coeffs.shape)))
            log.debug('[t=%d,n=%d] Init - S_x.shape           %s' % (self.t, self.n, str(self.S_x.shape)))
            log.debug('[t=%d,n=%d] Init - S_P.shape           %s' % (self.t, self.n, str(self.S_P.shape)))
            log.debug('[t=%d,n=%d] Init - fPos.shape          %s' % (self.t, self.n, str(self.fPositDerivSpike.shape)))
            log.debug('[t=%d,n=%d] Init - fNeg.shape          %s' % (self.t, self.n, str(self.fNegatDerivSpike.shape)))
            #log.debug('[t=%d,n=%d] Init - kalmanTh.shape      %s' % (self.t, self.n, str(self.kalmThreshold.shape)))
            return 1

        # For any other vol, if still a discard volume
        if self.n == 0:
            self.Data_FromAFNI = np.append(self.Data_FromAFNI,this_t_data[:, np.newaxis], axis=1)
            if self.save_ema: self.Data_EMA      = np.append(self.Data_EMA,    np.zeros((self.Nv,1)), axis=1)
            self.Data_iGLM     = np.append(self.Data_iGLM,   np.zeros((self.Nv,1)), axis=1)
            self.Data_kalman   = np.append(self.Data_kalman, np.zeros((self.Nv,1)), axis=1)
            if self.save_smooth: self.Data_smooth   = np.append(self.Data_smooth, np.zeros((self.Nv,1)), axis=1)
            self.Data_norm     = np.append(self.Data_norm,   np.zeros((self.Nv,1)), axis=1)
            self.Data_wind     = np.append(self.Data_wind,   np.zeros((self.Nv,1)), axis=1)
            self.nuisance      = np.append(self.nuisance,    np.zeros((self.iGLM_num_regressors,1)), axis=1)
            self.iGLM_Coeffs   = np.append(self.iGLM_Coeffs, np.zeros((self.Nv,self.iGLM_num_regressors,1)), axis=2)
            self.S_x           = np.append(self.S_x,         np.zeros((self.Nv,1)),   axis=1)
            self.S_P           = np.append(self.S_P,         np.zeros((self.Nv,1)),   axis=1)
            self.fPositDerivSpike = np.append(self.fPositDerivSpike,         np.zeros((self.Nv,1)),   axis=1)
            self.fNegatDerivSpike = np.append(self.fNegatDerivSpike,         np.zeros((self.Nv,1)),   axis=1)
            #self.kalmThreshold    = np.append(self.kalmThreshold,         np.zeros((self.Nv,1)),   axis=1)
            self.hits      = np.append(self.hits,      np.zeros((self.Ncaps,1)),  axis=1)
            self.svrscores = np.append(self.svrscores, np.zeros((self.Ncaps,1)), axis=1)
            log.debug('[t=%d,n=%d] Discard - Data_FromAFNI.shape %s' % (self.t, self.n, str(self.Data_FromAFNI.shape)))
            if self.save_ema: log.debug('[t=%d,n=%d] Discard - Data_EMA.shape      %s' % (self.t, self.n, str(self.Data_EMA.shape)))
            log.debug('[t=%d,n=%d] Discard - Data_iGLM.shape     %s' % (self.t, self.n, str(self.Data_iGLM.shape)))
            log.debug('[t=%d,n=%d] Discard - Data_norm.shape     %s' % (self.t, self.n, str(self.Data_norm.shape)))
            log.debug('[t=%d,n=%d] Discard - Data_wind.shape     %s' % (self.t, self.n, str(self.Data_wind.shape)))
            log.debug('[t=%d,n=%d] Discard - nuisance.shape      %s' % (self.t, self.n, str(self.nuisance.shape)))
            log.debug('[t=%d,n=%d] Discard - iGLM_Coeffs.shape   %s' % (self.t, self.n, str(self.iGLM_Coeffs.shape)))
            log.debug('[t=%d,n=%d] Discard - S_x.shape           %s' % (self.t, self.n, str(self.S_x.shape)))
            log.debug('[t=%d,n=%d] Discard - S_P.shape           %s' % (self.t, self.n, str(self.S_P.shape)))
            log.debug('[t=%d,n=%d] Discard - fPos.shape          %s' % (self.t, self.n, str(self.fPositDerivSpike.shape)))
            log.debug('[t=%d,n=%d] Discard - fNeg.shape          %s' % (self.t, self.n, str(self.fNegatDerivSpike.shape)))
            #log.debug('[t=%d,n=%d] Discard - kalmanTh.shape      %s' % (self.t, self.n, str(self.kalmThreshold.shape)))
            return 1

        # If we reach this point, it means we have work to do
        self.Data_FromAFNI = np.append(self.Data_FromAFNI,this_t_data[:, np.newaxis], axis=1)
        log.debug('[t=%d,n=%d] Online - Input - Data_FromAFNI.shape %s' % (self.t, self.n, str(self.Data_FromAFNI.shape)))

        # Do EMA (if needed)
        # ==================
        ema_data_out, self.EMA_filt = rt_EMA_vol(self.n, self.t, self.EMA_th, self.Data_FromAFNI, self.EMA_filt, do_operation = self.do_EMA)
        if self.save_ema:
            self.Data_EMA = np.append(self.Data_EMA, ema_data_out, axis=1)
            log.debug('[t=%d,n=%d] Online - EMA - Data_EMA.shape      %s' % (self.t, self.n, str(self.Data_EMA.shape)))
        
        # Do iGLM (if needed)
        # ===================
        if self.iGLM_motion:
            this_t_nuisance = np.concatenate((self.legendre_pols[self.t,:],motion))[:,np.newaxis]
        else:
            this_t_nuisance = (self.legendre_pols[self.t,:])[:,np.newaxis]
        self.nuisance = np.append(self.nuisance, this_t_nuisance, axis=1)
        log.debug('[t=%d,n=%d] Online - iGLM - nuisance.shape      %s' %  (self.t, self.n, str(self.nuisance.shape)))
        iGLM_data_out, self.iGLM_prev, Bn = rt_regress_vol(self.n, 
                                                           ema_data_out[:,np.newaxis],
                                                           self.nuisance[:,self.t][:,np.newaxis],
                                                           self.iGLM_prev,
                                                           do_operation = self.do_iGLM)
        self.Data_iGLM    = np.append(self.Data_iGLM, iGLM_data_out, axis=1)
        self.iGLM_Coeffs  = np.append(self.iGLM_Coeffs, Bn, axis = 2) 
        log.debug('[t=%d,n=%d] Online - iGLM - Data_iGLM.shape     %s' % (self.t, self.n, str(self.Data_iGLM.shape)))
        log.debug('[t=%d,n=%d] Online - iGLM - iGLM_Coeffs.shape   %s' % (self.t, self.n, str(self.iGLM_Coeffs.shape)))

        # Do Kalman Low-Pass Filter (if needed)
        # =====================================
        klm_data_out, Sx_out, SP_out, fPos_out, fNeg_out = rt_kalman_vol(self.n,
                                                                        self.t,
                                                                        self.Data_iGLM,
                                                                        self.S_x[:,self.t - 1],
                                                                        self.S_P[:,self.t - 1],
                                                                        self.fPositDerivSpike[:, self.t - 1],
                                                                        self.fNegatDerivSpike[:, self.t - 1],
                                                                        self.n_cores,
                                                                        self.nvols_discard,
                                                                        self.pool,
                                                                        do_operation = self.do_kalman)
        
        self.Data_kalman      = np.append(self.Data_kalman, klm_data_out, axis = 1)
        self.S_x              = np.append(self.S_x, Sx_out, axis=1)
        self.S_P              = np.append(self.S_P, SP_out, axis=1)
        self.fPositDerivSpike = np.append(self.fPositDerivSpike, fPos_out, axis=1)
        self.fNegatDerivSpike = np.append(self.fNegatDerivSpike, fNeg_out, axis=1)
        log.debug('[t=%d,n=%d] Online - iGLM - Data_kalman.shape     %s' % (self.t, self.n, str(self.Data_kalman.shape)))

        # Do Spatial Smoothing (if needed)
        # ================================
        smooth_out       = rt_smooth_vol(self.Data_kalman[:,self.t], self.mask_img, fwhm = self.FWHM, do_operation = self.do_smooth)
        if self.save_smooth:
            self.Data_smooth = np.append(self.Data_smooth, smooth_out, axis=1)
            log.debug('[t=%d,n=%d] Online - Smooth - Data_smooth.shape   %s' % (self.t, self.n, str(self.Data_smooth.shape)))

        # Do Spatial Normalization (if needed)
        # ====================================
        norm_out       = rt_snorm_vol(smooth_out, do_operation=self.do_snorm)
        self.Data_norm = np.append(self.Data_norm, norm_out, axis=1)
        log.debug('[t=%d,n=%d] Online - Smooth - Data_norm.shape   %s' % (self.t, self.n, str(self.Data_norm.shape)))

        # Do Windowing (if needed)
        # ========================
        if (self.t >= self.start_vol) and self.hit_dowin:
            vols_in_win       = (np.arange(self.t-4,self.t)+1)[::-1]
            out_data_windowed = np.dot(self.Data_norm[:,vols_in_win], self.hit_win_weights)
            self.Data_wind    = np.append(self.Data_wind, out_data_windowed, axis = 1)
        else:
            self.Data_wind    = np.append(self.Data_wind, norm_out, axis =1)

        # Compute SVR scores (if needed)
        # ==============================
        if self.t < self.start_vol:   # We don't want to start decoding before iGLM is stable.
            self.svrscores = np.append(self.svrscores, np.zeros((self.Ncaps,1)), axis=1)
        else:
            this_t_svrscores = rt_svrscore_vol(self.Data_wind[:, self.t], self.SVRs, self.caps_labels)
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
            log.info('[t=%d,n=%d] CAP hit [%s]' % (self.t,self.n, hit))
            self.hits[self.caps_labels.index(hit),self.t] = 1

        # Need to return something, otherwise the program thinks the experiment ended
        return 1
    
    def final_steps(self):
        if self.mask_img is None:
            log.warning(' final_steps = No outputs generated due to lack of mask.')
            return 1
        
        log.debug(' final_steps - About to write outputs to disk.')
        out_vars   = [self.Data_norm, self.Data_wind]
        out_labels = ['.pp_Zscore.nii','.pp_ExpWin.nii']
        if self.do_EMA:
            out_vars.append(self.Data_EMA)
            out_labels.append('.pp_EMA.nii')
        if self.do_iGLM:
            out_vars.append(self.Data_iGLM)
            out_labels.append('.pp_iGLM.nii')
        if self.do_kalman:
            out_vars.append(self.Data_kalman)
            out_labels.append('.pp_LPfilter.nii')
        if self.do_smooth:
            out_vars.append(self.Data_smooth)
            out_labels.append('.pp_Smooth.nii')
        for variable, file_suffix in zip(out_vars, out_labels):
            out = unmask_fMRI_img(variable, self.mask_img, osp.join(self.out_dir,self.out_prefix+file_suffix))

        if self.do_iGLM:
            for i,lab in enumerate(self.nuisance_labels):
                data = self.iGLM_Coeffs[:,i,:]
                out = unmask_fMRI_img(data, self.mask_img, osp.join(self.out_dir,self.out_prefix+'.pp_iGLM_'+lab+'.nii'))    

        np.save(osp.join(self.out_dir,self.out_prefix+'.svrscores'),self.svrscores)
        log.info('Saved svrscores to %s' % osp.join(self.out_dir,self.out_prefix+'.svrscores'))
        np.save(osp.join(self.out_dir,self.out_prefix+'.hits'), self.hits)
        log.info('Saved hits info to %s' % osp.join(self.out_dir,self.out_prefix+'.hits'))
        return 1

def processExperimentOptions (self, options=None):

    """
       Process command line options for on-going experiment.
       Customize as needed for your own experiments.
    """

    usage = "%prog [options]"
    description = "AFNI real-time demo receiver with demo visualization."
    parser = OptionParser(usage=usage, description=description)
    parser.add_option("-d", "--debug", action="store_true",
             help="enable debugging output")
    parser.add_option("-v", "--verbose", action="store_true",
             help="enable verbose output")
    parser.add_option("-p", "--tcp_port", help="TCP port for incoming connections")
    parser.add_option("-S", "--show_data", action="store_true",
            help="display received data in terminal if this option is specified")
    
    parser.add_option("--no_ema",    help="De-activate EMA Filtering Step",              dest="do_EMA",      default=True, action="store_false")
    parser.add_option("--no_iglm",   help="De-activate iGLM Denoising Step",             dest="do_iGLM",     default=True, action="store_false")
    parser.add_option("--no_kalman", help="De-activate Kalman Low-Pass Filter Step",     dest="do_kalman",   default=True, action="store_false")
    parser.add_option("--no_smooth", help="De-activate Spatial Smoothing Step",          dest="do_smooth",   default=True, action="store_false")
    parser.add_option("--no_snorm",  help="De-activate per-volume spartial Z-Scoring",   dest="do_snorm",   default=True, action="store_false")
    parser.add_option("--fwhm",      help="FWHM for Spatial Smoothing in [mm]",          dest="FWHM",        default=4.0, action="store", type="float")
    parser.add_option("--polort",     help="Order of Legengre Polynomials for iGLM",     dest="iGLM_polort", default=2, action="store", type="int")
    parser.add_option("--no_iglm_motion", help="Do not use 6 motion parameters in iGLM", dest="iGLM_motion", default=True, action="store_false")
    parser.add_option("--discard",    help="Number of volumes to discard (they won't enter the iGLM step)",  default=10, dest="discard", action="store", type="int")
    parser.add_option("--start",      help="Volume when decoding should start. When we think iGLM is sufficient_stable", default=100, dest="start_vol", action="store", type="int")
    parser.add_option("--nvols",      help="Number of expected volumes (for legendre pols only)", dest="nvols",default=500, action="store", type="int")
    parser.add_option("--tr",         help="Repetition time [sec]", dest="tr",default=1.0, action="store", type="float")
    parser.add_option("--ncores",     help="Number of cores to use in the parallel processing part of the code", dest="n_cores", action="store",type="int", default=8)
    parser.add_option("--mask",       help="Mask necessary for smoothing operation", dest="mask_path", action="store", type="str", default=None)
    parser.add_option("--out_dir",    help="Output directory",   dest="out_dir",    action="store", type="str", default="./")
    parser.add_option("--out_prefix", help="Prefix for outputs", dest="out_prefix", action="store", type="str", default="online_preproc")
    parser.add_option("--svr_path",   help="Path to pre-trained SVR models", dest="svr_path", action="store", type="str", default=None)
    parser.add_option("--no_fscreen", help="Do not use full screen", dest="fscreen", action="store_false", default=True) 
    parser.add_option("--save_ema",    help="Save 4D EMA dataset",     dest="save_ema",   default=False, action="store_true")
    parser.add_option("--save_smooth", help="Save 4D Smooth dataset",     dest="save_smooth",   default=False, action="store_true")

    return parser.parse_args(options)

def main():
    # 1) Read Input Parameters: port, fullscreen, etc..
    log.info('1) Reading input parameters...')
    opts, args = processExperimentOptions(sys.argv)
    log.debug('User Options: %s' % str(opts))    
    # 2) Create Experiment Object
    log.info('2) Instantiating Experiment Object...')
    experiment = Experiment(opts)

    # 3) Initiate GUI (Psychopy Screen)
    log.info('3) Initializing Psychopy Experiment...')
    experiment.setup_experiment(opts)

    # 2) Start Communications
    log.info('4) Opening Communication Channel...')
    receiver = ReceiverInterface(port=opts.tcp_port, show_data=opts.show_data)
    if not receiver:
        return 1

    # set signal handlers and look for data
    log.info('5) Setting Signal Handlers...')
    receiver.set_signal_handlers()  # require signal to exit

    # set receiver callback
    # At this point Receiver is still basically an empty container
    receiver.compute_TR_data  = experiment.compute_TR_data
    receiver.final_steps      = experiment.final_steps

    # prepare for incoming connections
    log.info('6) Prepare for Incoming Connections...')
    if receiver.RTI.open_incoming_socket():
        return 1

    # # repeatedly: process all incoming data for a single run
    # while 1:
    #     rv = receiver.process_one_run()
    #     if rv: time.sleep(1)              # on error, ponder life briefly
    #     receiver.close_data_ports()
    # return -1   
    #Vinai's alternative
    log.info('7) Here we go...')
    rv = receiver.process_one_run()
    return rv


if __name__ == '__main__':
   sys.exit(main())
