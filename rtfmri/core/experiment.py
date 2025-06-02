import sys
import os.path as osp
from types import SimpleNamespace

import numpy as np
import pandas as pd

from core.pipeline import Pipeline
from matching.matcher import SVRMatcher
from matching.hit_detector import HitDetector
from utils.log import set_logger

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))
sys.path.append('..')

from utils.fMRI import load_fMRI_file, unmask_fMRI_img


class Experiment:
    """
    Class representing a real-time fMRI experiment.

    This class handles the setup of the experiment, initialization of the preprocessing
    pipeline, and the management of incoming data from the scanner.

    Parameters
    ----------
    options : Options
        Configuration object containing experiment parameters (e.g., TR, number of volumes, paths).
    mp_evt_hit : multiprocessing.Event
        Event used to signal a CAP hit.
    mp_evt_end : multiprocessing.Event
        Event used to signal the end of the experiment.
    mp_evt_qa_end : multiprocessing.Event
        Event used to signal the end of a QA block.
    """
    def __init__(self, options, mp_evt_hit, mp_evt_end, mp_evt_qa_end):
        self.log = set_logger(options.debug, options.silent)

        self.mp_evt_hit = mp_evt_hit # Signals a CAP hit
        self.mp_evt_end = mp_evt_end # Signals the end of the experiment
        self.mp_evt_qa_end = mp_evt_qa_end # Signals the end of a QA set

        self.ewin = None
        self.exp_type = options.exp_type
        self.no_proc_chair = options.no_proc_chair
        self.screen_size = [512, 288]
        self.fullscreen = options.fullscreen
        self.screen = options.screen

        self.n = 0 # Counter for number of volumes pre-processed (Start = 1)
        self.t = -1 # Counter for number of received volumes (Start = 0
        self.Nv= None # Number of voxels in data mask
        self.Nt = options.nvols # Number acquisitions
        self.TR = options.tr # TR [seconds]

        self.nvols_discard = options.discard # Number of volumes to discard from any analysis (won't enter pre-processing)

        if options.mask_path is None:
            self.log.warning('  Experiment_init_ - No mask was provided!')
            self.mask_img = None
        else:
            self.mask_img  = load_fMRI_file(options.mask_path)
            self.mask_Nv = int(np.sum(self.mask_img.get_fdata()))
            self.log.debug('  Experiment_init_ - Number of Voxels in user-provided mask: %d' % self.mask_Nv)

        self.pipe = Pipeline(options, self.Nt, self.mask_Nv, self.mask_img, self.exp_type)        

    def _compute_TR_data_impl(self, motion, extra):
        """
        Process data for the current TR by passing it to the pipeline.

        Parameters
        ----------
        motion : list of list[float]
            List of 6-element motion parameter lists (one per TR).
        extra : list of list[float]
            List of voxel values for the current TR, where each sublist contains time series data for a voxel.

        Returns
        -------
        np.array
            pipeline.processed_tr (the processed data for this TR)
        """
        self.t += 1
        self.log.info(f' - Time point [t={self.t}, n={self.n}]')

        # Keep a record of motion estimates
        motion = [i[self.t] for i in motion]
        self.pipe.motion_estimates.append(motion)

        if len(motion) != 6:
            self.log.error('Motion not read in correctly.')
            self.log.error(f'Expected length: 6 | Actual length: {len(motion)}')
            sys.exit(-1)
        
        this_t_data = np.array([e[self.t] for e in extra])
        del extra # Save resources

        self.Nv = len(this_t_data)

        if self.t > 0:
            if len(this_t_data) != self.Nv:
                self.log.error(f'Extra data not read in correctly.')
                self.log.error(f'Expected length: {self.Nv} | Actual length: {len(this_t_data)}')
                sys.exit(-1)
        
        # Update n (only if not longer a discard volume)
        if self.t > self.nvols_discard - 1:
            self.n += 1

        return self.pipe.process(self.t, self.n, motion, this_t_data)

    def compute_TR_data(self, motion, extra):
        _ = self._compute_TR_data_impl(motion, extra)
        return 1
    
    def end_run(self, save=True):
        """Finalize the experiment by saving all outputs and signaling completion."""
        if save:
            self.pipe.final_steps()
        self.pipe.final_steps()
        self.mp_evt_end.set()


class ESAMExperiment(Experiment):
    """
    Class for running a real-time fMRI experiment in Experience Sampling (ESAM) mode.

    This class extends `Experiment` to support online template matching and GUI presentation.

    Attributes
    ----------
    lastQA_endTR : int
        The TR index of the last time a QA block ended.
    
    vols_noqa : int
        Number of volumes to skip after QA ends before hit detection resumes.

    outhtml : str
        Path to the dynamic HTML report output.

    qa_onsets : list of int
        List of TRs where QA blocks began.

    qa_offsets : list of int
        List of TRs where QA blocks ended.

    qa_onsets_path : str
        Path where QA onsets will be saved.

    qa_offsets_path : str
        Path where QA offsets will be saved.

    matcher : SVRMatcher
        Object that performs template matching with SVR models.

    hits : np.ndarray
        2D array tracking detected hits [template x time].

    hit_detector : HitDetector
        Object that applies hit detection logic to matcher scores.
    """
    def __init__(self, options, mp_evt_hit, mp_evt_end, mp_evt_qa_end):
        super().__init__(options, mp_evt_hit, mp_evt_end, mp_evt_qa_end)
        self.lastQA_endTR  = 0
        self.out_dir = options.out_dir
        self.out_prefix = options.out_prefix
        self.outhtml = osp.join(self.out_dir,self.out_prefix+'.dyn_report')

        self.qa_onsets = []
        self.qa_offsets = []
        self.qa_onsets_path  = osp.join(self.out_dir,self.out_prefix+'.qa_onsets.txt')
        self.qa_offsets_path = osp.join(self.out_dir,self.out_prefix+'.qa_offsets.txt')
        
        # Convert dicts into a objects that allow dot notation (ex. matching_opts.matcher_type)
        matching_opts = SimpleNamespace(**options.matching)
        hit_opts = SimpleNamespace(**options.hits)

        self.vols_noqa = matching_opts.vols_noqa
        
        # TODO: add in other matcher options as I make them
        if matching_opts.matcher_type == "svr":
            self.matcher = SVRMatcher(matching_opts)

        self.hits = np.zeros((self.matcher.Ntemplates, 1))
        self.hit_detector = HitDetector(hit_opts)
        
        
    def compute_TR_data(self, motion, extra):
        hit_status    = self.mp_evt_hit.is_set()
        qa_end_status = self.mp_evt_qa_end.is_set()

        processed = super()._compute_TR_data_impl(motion, extra)
        scores = self.matcher.match(self.t, self.n, processed)

        if qa_end_status:
            self.lastQA_endTR = self.t
            self.qa_offsets.append(self.t)
            self.mp_evt_qa_end.clear()
            self.log.info(f'QA ended (cleared) --> updating lastQA_endTR = {self.lastQA_endTR}')
        
        if hit_status or (self.t <= self.lastQA_endTR + self.vols_noqa):
            hit = None
        else:
            hit = self.hit_detector.detect(self.t, self.matcher.template_labels, scores)
        
        if hit:
            self.log.info(f'[t=self.t,n=self.n] =============================================  CAP hit [hit]')
            self.qa_onsets.append(self.t)
            self.hits[self.caps_labels.index(hit),self.t] = 1
            self.mp_evt_hit.set()
            
        return 1

    def end_run(self, save=True):
        if save:
            self.pipe.final_steps()
        self.mp_evt_end.set()
