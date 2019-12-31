#!/usr/bin/env python

# python3 status: compatible

import sys, os
import itertools
import signal, time
#from   optparse import OptionParser
import argparse
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import numpy as np
import multiprocessing
#multiprocessing.set_start_method('spawn', True)
import os.path as osp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from afni_lib.receiver import ReceiverInterface
#from afni_lib.receiver import ReceiverInterface
from rtcap_lib.core import unpack_extra, welford
from rtcap_lib.rt_functions import rt_EMA_vol, rt_regress_vol, rt_kalman_vol, rt_smooth_vol, rt_snorm_vol
from rtcap_lib.rt_functions import gen_polort_regressors
from rtcap_lib.fMRI import load_fMRI_file, unmask_fMRI_img

# ----------------------------------------------------------------------
# Default Login Options:
log     = logging.getLogger("online_preproc")
log_fmt = logging.Formatter('[%(levelname)s - Main]: %(message)s')
log_ch  = logging.StreamHandler()
log_ch.setFormatter(log_fmt)
#log_ch.setLevel(logging.INFO)
log.setLevel(logging.INFO)
log.addHandler(log_ch)

g_help_string = """`
=============================================================================
online_preprocessing.py - program for online preprocessing data provided via 
TCP connection. Included pre-processing steps: 1) EMA Filter, Incremental GLM,
Kalman Low-Pass Filter, Spatial Smoothing.

------------------------------------------
   Options:
   
=============================================================================
"""
g_history = """
   0.1  Dec 12, 2019 : added support for pre-processing online
"""

g_version = "online_preprocessing.py version 1.0, December 12, 2019"

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

    def compute_TR_data(self, motion, extra):
        # Keep a record of motion estimates
        #print(motion)
        #print(type(motion))
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
            return 1

        # Compute running mean and running std with welford
        self.welford_M, self.welford_S, self.weldord_std = welford(self.n, this_t_data, self.welford_M, self.welford_S)
        #log.debug('[t=%d,n=%d] Online - Input - welford_M.shape %s' % (self.t, self.n, str(self.welford_M.shape)))
        #log.debug('[t=%d,n=%d] Online - Input - welford_S.shape %s' % (self.t, self.n, str(self.welford_S.shape)))
        #log.debug('[t=%d,n=%d] Online - Input - welford_std.shape %s' % (self.t, self.n, str(self.welford_std.shape)))

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
        #log.debug('[t=%d,n=%d] ========================   KALMAN   ==================================================')
        #log.debug('[t=%d,n=%d] Online - Kalman_PRE - Data_iGLM           %s' % (self.t, self.n, str(self.Data_iGLM.shape)))
        #log.debug('[t=%d,n=%d] Online - Kalman_PRE - S_x[:,self.t - 1]   %s' % (self.t, self.n, str(self.S_x.shape)))
        #log.debug('[t=%d,n=%d] Online - Kalman_PRE - S_P[:,self.t - 1]   %s' % (self.t, self.n, str(self.S_P.shape)))
        #log.debug('[t=%d,n=%d] Online - Kalman_PRE - fPos[:, self.t - 1] %s' % (self.t, self.n, str(self.fPositDerivSpike.shape)))
        #log.debug('[t=%d,n=%d] Online - Kalman_PRE - fNeg[:, self.t - 1] %s' % (self.t, self.n, str(self.fNegatDerivSpike.shape)))

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
        
        
        #log.debug('[t=%d,n=%d] Online - Kalman_POST - klm_data_out %s' % (self.t, self.n, str(klm_data_out.shape)))
        #log.debug('[t=%d,n=%d] Online - Kalman_POST - S_x          %s' % (self.t, self.n, str(self.S_x.shape)))
        #log.debug('[t=%d,n=%d] Online - Kalman_POST - S_P          %s' % (self.t, self.n, str(self.S_P.shape)))
        #log.debug('[t=%d,n=%d] Online - Kalman_POST - fPos         %s' % (self.t, self.n, str(self.fPositDerivSpike.shape)))
        #log.debug('[t=%d,n=%d] Online - Kalman_POST - fNeg         %s' % (self.t, self.n, str(self.fNegatDerivSpike.shape)))
        
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
        #log.debug('[t=%d,n=%d] Online - Smooth - Data_norm.shape   %s' % (self.t, self.n, str(self.Data_norm.shape)))
        # Need to return something, otherwise the program thinks the experiment ended
        return 1

    def final_steps(self):
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

        

        return 1
def processExperimentOptions (self, options=None):
    parser = argparse.ArgumentParser(description="rtCAPs experimental software. Based on NIH-neurofeedback software")
    parser_gen = parser.add_argument_group("General Options")
    parser_gen.add_argument("-d", "--debug", action="store_true", dest="debug",  help="Enable debugging output [%(default)s]", default=False)
    parser_gen.add_argument("-s", "--silent",   action="store_true", dest="silent", help="Minimal text messages [%(default)s]", default=False)
    parser_gen.add_argument("-p", "--tcp_port", help="TCP port for incoming connections [%(default)s]", action="store", default=53214, type=int, dest='tcp_port')
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
    parser_exp.add_argument("-e","--exp_type", help="Type of Experimental Run [%(default)s]",      type=str, required=True,  choices=['proc','esam'], default='proc')
    parser_exp.add_argument("--no_proc_chair", help="Hide crosshair during preprocessing run [%(default)s]", default=False,  action="store_true", dest='no_proc_chair')
    
    return parser.parse_args(options)
# def processExperimentOptions (self, options=None):

#     """
#        Process command line options for on-going experiment.
#        Customize as needed for your own experiments.
#     """

#     usage = "%prog [options]"
#     description = "AFNI real-time demo receiver with demo visualization."
#     parser = OptionParser(usage=usage, description=description)
##     parser.add_option("-d", "--debug",    action="store_true", dest="debug",  help="enable debugging output",          default=False)
##     parser.add_option("-s", "--silent",   action="store_true", dest="silent", help="make program do minimal printing", default=False)
    
##     parser.add_option("-p", "--tcp_port", help="TCP port for incoming connections")
#     parser.add_option("-S", "--show_data", action="store_true",
#             help="display received data in terminal if this option is specified")
    
##     parser.add_option("--no_ema",    help="De-activate EMA Filtering Step [default: %default]",              dest="do_EMA",      default=True, action="store_false")
##     parser.add_option("--no_iglm",   help="De-activate iGLM Denoising Step  [default: %default]",             dest="do_iGLM",     default=True, action="store_false")
##     parser.add_option("--no_kalman", help="De-activate Kalman Low-Pass Filter Step  [default: %default]",     dest="do_kalman",   default=True, action="store_false")
##     parser.add_option("--no_smooth", help="De-activate Spatial Smoothing Step  [default: %default]",          dest="do_smooth",   default=True, action="store_false")
##     parser.add_option("--no_snorm",  help="De-activate per-volume spartial Z-Scoring  [default: %default]",   dest="do_snorm",   default=True, action="store_false")
#     parser.add_option("--fwhm",      help="FWHM for Spatial Smoothing in [mm]  [default: %default]",          dest="FWHM",        default=4.0, action="store", type="float")
##     parser.add_option("--polort",     help="Order of Legengre Polynomials for iGLM  [default: %default]",     dest="iGLM_polort", default=2, action="store", type="int")
##     parser.add_option("--no_iglm_motion", help="Do not use 6 motion parameters in iGLM  [default: %default]", dest="iGLM_motion", default=True, action="store_false")
##     parser.add_option("--discard",    help="Number of volumes to discard (they won't enter the iGLM step)  [default: %default]",  default=10, dest="discard", action="store", type="int")
##     parser.add_option("--nvols",      help="Number of expected volumes (for legendre pols only)  [default: %default]", dest="nvols",default=500, action="store", type="int")
##     parser.add_option("--tr",         help="Repetition time [sec]  [default: %default]",                      dest="tr",default=1.0, action="store", type="float")
##     parser.add_option("--ncores",     help="Number of cores to use in the parallel processing part of the code  [default: %default]", dest="n_cores", action="store",type="int", default=10)
##    parser.add_option("--mask",       help="Mask necessary for smoothing operation  [default: %default]",     dest="mask_path", action="store", type="str", default=None)
#     parser.add_option("--out_dir",    help="Output directory  [default: %default]",                           dest="out_dir",    action="store", type="str", default="./")
#     parser.add_option("--out_prefix", help="Prefix for outputs  [default: %default]",                         dest="out_prefix", action="store", type="str", default="online_preproc")
#     parser.add_option("--save_ema",    help="Save 4D EMA dataset  [default: %default]",     dest="save_ema",   default=False, action="store_true")
#     parser.add_option("--save_kalman", help="Save 4D Smooth dataset  [default: %default]",     dest="save_kalman",   default=False, action="store_true")
#     parser.add_option("--save_smooth", help="Save 4D Smooth dataset  [default: %default]",     dest="save_smooth",   default=False, action="store_true")
#     parser.add_option("--save_iglm  ", help="Save 4D iGLM datasets  [default: %default]",     dest="save_iglm",   default=False, action="store_true")
#     parser.add_option("--save_orig"  , help="Save 4D with incoming data  [default: %default]", dest="save_orig", default=False, action="store_true")
#     parser.add_option("--save_all"  , help="Save 4D with incoming data  [default: %default]", dest="save_all", default=False, action="store_true")

#     return parser.parse_args(options)

def main():
    # 1) Read Input Parameters: port, fullscreen, etc..
    log.info('1) Reading input parameters...')
    opts, args = processExperimentOptions(sys.argv)
    log.debug('User Options: %s' % str(opts))    
    # 2) Create Experiment Object
    log.info('2) Instantiating Experiment Object...')
    experiment = Experiment(opts)

    # 2) Start Communications
    log.info('3) Opening Communication Channel...')
    receiver = ReceiverInterface(port=opts.tcp_port, show_data=opts.show_data)
    if not receiver:
        return 1

    # set signal handlers and look for data
    log.info('4) Setting Signal Handlers...')
    receiver.set_signal_handlers()  # require signal to exit

    # set receiver callback
    # At this point Receiver is still basically an empty container
    receiver.compute_TR_data  = experiment.compute_TR_data
    receiver.final_steps      = experiment.final_steps

    # prepare for incoming connections
    log.info('5) Prepare for Incoming Connections...')
    if receiver.RTI.open_incoming_socket():
        return 1

    #Vinai's alternative
    log.info('6) Here we go...')
    rv = receiver.process_one_run()
    return rv


if __name__ == '__main__':
   sys.exit(main())
