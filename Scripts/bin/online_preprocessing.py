#!/usr/bin/env python

# python3 status: compatible

import sys, os
import signal, time
from   optparse import OptionParser
import logging as log
import numpy as np

sys.path.append("../")
from afni_lib import ReceiverInterface
from rtcap_lib.core import unpack_extra
from rtcap_lib.rt_functions import rt_EMA_vol, rt_regress_vol, gen_polort_regressors
# ----------------------------------------------------------------------
# globals
#log = logging.getLogger(__name__)
#log.setLevel(logging.DEBUG)
log.basicConfig(format='[%(levelname)s]: %(message)s', level=log.DEBUG)
g_help_string = """
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
        self.n             = 0              # Counter for number of volumes pre-processed (Start = 1)
        self.t             = -1             # Counter for number of received volumes (Start = 0)
        self.Nv            = None           # Number of voxels in data mask
        self.Nt            = options.nvols  # Number acquisitions
        self.TR            = options.tr     # TR [seconds]
        self.Data_FromAFNI = None           # np.array [Nv,Nt] for incoming data
        self.Data_EMA      = None           # np.array [Nv,Nt] for data after EMA  step
        self.Data_iGLM     = None           # np.array [Nv,Nt] for data after iGLM step
        self.Data_kalma    = None           # np.array [Nv,Nt] for data after low-pass step
        self.Data_smooth   = None           # np.array [Nv,Nt] for data after spatial smoothing
        self.Data_norm     = None           # np.array [Nv,Nt] for data after spatial normalization (spatial z-score)
        self.iGLM_Coeffs   = None           # np.array [Nregressor, Nv, Nt] for beta coefficients for all regressors
        
        self.do_EMA        = options.do_EMA       # Should we do EMA
        self.do_iGLM       = options.do_iGLM      # Should we do iGLM
        self.do_kalman     = options.do_kalman    # Should we do Low Pass Filter
        self.do_smooth     = options.do_smooth    # Should we do Spatial Filtering
        self.FWHM          = options.FWHM         # FWHM for Spatial Smoothing in [mm]
        
        self.nvols_discard = options.discard      # Number of volumes to discard from any analysis (won't enter pre-processing)

        self.iGLM_prev     = {}
        self.iGLM_motion   = options.iGLM_motion
        self.iGLM_polort   = options.iGLM_polort
        self.nuisance      = None
        if self.iGLM_motion:
            self.iGLM_num_regressors = self.iGLM_polort + 6
        else:
            self.iGLM_num_regressors = self.iGLM_polort

        self.S_x              = None
        self.S_P              = None
        self.fPositDerivSpike = None
        self.fNegatDerivSpike = None
        self.kalmThreshold    = None

        self.EMA_th   = 0.98
        self.EMA_filt = None
        

        # Create Legendre Polynomial regressors
        if self.iGLM_polort > -1:
            self.legendre_pols = gen_polort_regressors(self.iGLM_polort,self.Nt)
        else:
            self.legendre_pols = None


        
    def compute_TR_data(self, motion, extra):
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
            self.Data_FromAFNI = np.array(this_t_data[:,np.newaxis])
            self.Data_EMA      = np.zeros((self.Nv,1))
            self.Data_iGLM     = np.zeros((self.Nv,1))
            self.Data_kalman   = np.zeros((self.Nv,1))
            self.Data_smooth   = np.zeros((self.Nv,1))
            self.Data_norm     = np.zeros((self.Nv,1))
            self.nuisance      = np.zeros((self.iGLM_num_regressors,1))
            self.iGLM_Coeffs   = np.zeros((self.Nv,self.iGLM_num_regressors,1))
            log.debug('[t=%d,n=%d] Init - Data_FromAFNI.shape %s' % (self.t, self.n, str(self.Data_FromAFNI.shape)))
            log.debug('[t=%d,n=%d] Init - Data_EMA.shape      %s' % (self.t, self.n, str(self.Data_EMA.shape)))
            log.debug('[t=%d,n=%d] Init - Data_iGLM.shape     %s' % (self.t, self.n, str(self.Data_iGLM.shape)))
            log.debug('[t=%d,n=%d] Init - nuisance.shape      %s' % (self.t, self.n, str(self.nuisance.shape)))
            log.debug('[t=%d,n=%d] Init - iGLM_Coeffs.shape   %s' % (self.t, self.n, str(self.iGLM_Coeffs.shape)))
            return 1

        # For any other vol, if still a discard volume
        if self.n == 0:
            self.Data_FromAFNI = np.append(self.Data_FromAFNI,this_t_data[:, np.newaxis], axis=1)
            self.Data_EMA      = np.append(self.Data_EMA,    np.zeros((self.Nv,1)), axis=1)
            self.Data_iGLM     = np.append(self.Data_iGLM,   np.zeros((self.Nv,1)), axis=1)
            self.Data_kalman   = np.append(self.Data_kalman, np.zeros((self.Nv,1)), axis=1)
            self.Data_smooth   = np.append(self.Data_smooth, np.zeros((self.Nv,1)), axis=1)
            self.Data_norm     = np.append(self.Data_norm,   np.zeros((self.Nv,1)), axis=1)
            self.nuisance      = np.append(self.nuisance,    np.zeros((self.iGLM_num_regressors,1)), axis=1)
            self.iGLM_Coeffs   = np.append(self.iGLM_Coeffs, np.zeros((self.Nv,self.iGLM_num_regressors,1)), axis=2)
            log.debug('[t=%d,n=%d] Discard - Data_FromAFNI.shape %s' % (self.t, self.n, str(self.Data_FromAFNI.shape)))
            log.debug('[t=%d,n=%d] Discard - Data_EMA.shape      %s' % (self.t, self.n, str(self.Data_EMA.shape)))
            log.debug('[t=%d,n=%d] Discard - Data_iGLM.shape     %s' % (self.t, self.n, str(self.Data_iGLM.shape)))
            log.debug('[t=%d,n=%d] Discard - nuisance.shape      %s' % (self.t, self.n, str(self.nuisance.shape)))
            log.debug('[t=%d,n=%d] Discard - iGLM_Coeffs.shape   %s' % (self.t, self.n, str(self.iGLM_Coeffs.shape)))
            return 1

        # If we reach this point, it means we have work to do
        self.Data_FromAFNI = np.append(self.Data_FromAFNI,this_t_data[:, np.newaxis], axis=1)
        log.debug('[t=%d,n=%d] Online - Input - Data_FromAFNI.shape %s' % (self.t, self.n, str(self.Data_FromAFNI.shape)))

        # Do EMA (if needed)
        ema_data_out, self.EMA_filt = rt_EMA_vol(self.n, self.t, self.EMA_th, self.Data_FromAFNI, self.EMA_filt, do_operation = self.do_EMA)
        self.Data_EMA = np.append(self.Data_EMA, ema_data_out, axis=1)
        log.debug('[t=%d,n=%d] Online - EMA - Data_EMA.shape      %s' % (self.t, self.n, str(self.Data_EMA.shape)))
        
        # Do iGLM (if needed)
        if self.iGLM_motion:
            this_t_nuisance = np.concatenate((self.legendre_pols[self.t,:],motion))[:,np.newaxis]
        else:
            this_t_nuisance = (self.legendre_pols[self.t,:])[:,np.newaxis]
        self.nuisance = np.append(self.nuisance, this_t_nuisance, axis=1)
        log.debug('[t=%d,n=%d] Online - iGLM - nuisance.shape      %s' %  (self.t, self.n, str(self.nuisance.shape)))
        iGLM_data_out, self.iGLM_prev, Bn = rt_regress_vol(self.n, 
                                                           self.Data_EMA[:,self.t][:,np.newaxis],
                                                           self.nuisance[:,self.t][:,np.newaxis],
                                                           self.iGLM_prev,
                                                           do_operation = self.do_iGLM)
        self.Data_iGLM    = np.append(self.Data_iGLM, iGLM_data_out, axis=1)
        self.iGLM_Coeffs  = np.append(self.iGLM_Coeffs, Bn, axis = 2) 
        log.debug('[t=%d,n=%d] Online - iGLM - Data_iGLM.shape     %s' % (self.t, self.n, str(self.Data_iGLM.shape)))
        log.debug('[t=%d,n=%d] Online - iGLM - iGLM_Coeffs.shape   %s' % (self.t, self.n, str(self.iGLM_Coeffs.shape)))

        # Need to return something, otherwise the program thinks the experiment ended
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
    
    parser.add_option("--no_ema",    help="De-activate EMA Filtering Step",          dest="do_EMA",    default=True, action="store_false")
    parser.add_option("--no_iglm",   help="De-activate iGLM Denoising Step",         dest="do_iGLM",   default=True, action="store_false")
    parser.add_option("--no_kalman", help="De-activate Kalman Low-Pass Filter Step", dest="do_kalman", default=True, action="store_false")
    parser.add_option("--no_smooth", help="De-activate Spatial Smoothing Step",      dest="do_smooth", default=True, action="store_false")
    parser.add_option("--fwhm",   help="FWHM for Spatial Smoothing in [mm]",         dest="FWHM",      default=4.0, action="store", type="float")
    parser.add_option("--polort", help="Order of Legengre Polynomials for iGLM",     dest="iGLM_polort",default=2, action="store", type="int")
    parser.add_option("--no_iglm_motion", help="Do not use 6 motion parameters in iGLM",     dest="iGLM_motion",default=True, action="store_false")
    parser.add_option("--discard", help="Number of volumes to discard (they won't enter the iGLM step)",     dest="discard",default=10, action="store", type="int")
    parser.add_option("--nvols", help="Number of expected volumes (for legendre pols only)", dest="nvols",default=500, action="store", type="int")
    parser.add_option("--tr", help="Repetition time [sec]", dest="tr",default=1.0, action="store", type="float")
    
    return parser.parse_args(options)

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
    #receiver.final_steps      = demo.final_steps

    # prepare for incoming connections
    log.info('5) Prepare for Incoming Connections...')
    if receiver.RTI.open_incoming_socket():
        return 1

    # # repeatedly: process all incoming data for a single run
    # while 1:
    #     rv = receiver.process_one_run()
    #     if rv: time.sleep(1)              # on error, ponder life briefly
    #     receiver.close_data_ports()
    # return -1   
    #Vinai's alternative
    log.info('6) Here we go...')
    rv = receiver.process_one_run()
    return rv


if __name__ == '__main__':
   sys.exit(main())