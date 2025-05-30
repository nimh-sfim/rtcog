import sys
import os.path as osp
import numpy as np
import pandas as pd

from .pipeline import Pipeline
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
        self.mp_evt_end.set()
