import logging
import sys
import os.path as osp
import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas

from core.preproc import Pipeline
from utils.log import get_logger, set_logger
from paths import CAP_labels

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))
sys.path.append('..')

from utils.fMRI import load_fMRI_file, unmask_fMRI_img


class Experiment:
    def __init__(self, options, mp_evt_hit, mp_evt_end, mp_evt_qa_end):
        self.log = set_logger(options.debug, options.silent)

        self.mp_evt_hit = mp_evt_hit           # Signals a CAP hit
        self.mp_evt_end = mp_evt_end           # Signals the end of the experiment
        self.mp_evt_qa_end = mp_evt_qa_end     # Signals the end of a QA set

        self.ewin          = None
        self.exp_type      = options.exp_type
        self.no_proc_chair = options.no_proc_chair
        self.screen_size   = [512, 288]
        self.fullscreen    = options.fullscreen
        self.screen        = options.screen

        self.n             = 0               # Counter for number of volumes pre-processed (Start = 1)
        self.t             = -1              # Counter for number of received volumes (Start = 0
        self.lastQA_endTR  = 0
        self.vols_noqa     = options.vols_noqa
        self.Nv            = None            # Number of voxels in data mask
        self.Nt            = options.nvols   # Number acquisitions
        self.TR            = options.tr      # TR [seconds]

        self.nvols_discard = options.discard      # Number of volumes to discard from any analysis (won't enter pre-processing)

        if options.mask_path is None:
            self.log.warning('  Experiment_init_ - No mask was provided!')
            self.mask_img = None
        else:
            self.mask_img  = load_fMRI_file(options.mask_path)
            self.mask_Nv = int(np.sum(self.mask_img.get_fdata()))
            self.log.debug('  Experiment_init_ - Number of Voxels in user-provided mask: %d' % self.mask_Nv)

        self.pipe = Pipeline(options, self.Nt, self.mask_Nv, self.mask_img, self.exp_type)        

    def compute_TR_data(self, motion, extra):
        """Receive data from the scanner, then pass to Pipeline for processing"""
        self.t += 1

        # Keep a record of motion estimates
        motion = [i[self.t] for i in motion]
        self.pipe.motion_estimates.append(motion)

        if len(motion) != 6:
            self.log.error('Motion not read in correctly.')
            self.log.error(f'Expected length: 6 | Actual length: {len(motion)}')
            sys.exit(-1)
        
        this_t_data = np.array([e[self.t] for e in extra])
        
        self.Nv = len(this_t_data)

        if self.t > 0:
            if len(this_t_data) != self.Nv:
                self.log.error(f'Extra data not read in correctly.')
                self.log.error(f'Expected length: {self.Nv} | Actual length: {len(this_t_data)}')
                sys.exit(-1)
        
        # Update n (only if not longer a discard volume)
        if self.t > self.nvols_discard - 1:
            self.n += 1
        
        self.pipe.process(self.t, self.n, motion, this_t_data)
        
        del extra # Save resources
        
        return 1
    
    def end_run(self):
        self.pipe.final_steps()
        self.mp_evt_end.set()