import sys
import os.path as osp
from multiprocessing import Process
from multiprocessing import Value, Manager
from multiprocessing.shared_memory import SharedMemory
from ctypes import c_int

import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas

from rtfmri.preproc.pipeline import Pipeline
from rtfmri.preproc.step_types import StepType
from rtfmri.matching.matcher import Matcher
from rtfmri.matching.matching_opts import MatchingOpts
from rtfmri.matching.hit_opts import HitOpts
from rtfmri.matching.hit_detector import HitDetector
from rtfmri.viz.streaming import run_streamer
from rtfmri.viz.streaming_config import StreamingConfig
from rtfmri.utils.log import set_logger
from rtfmri.utils.fMRI import load_fMRI_file, unmask_fMRI_img


class Experiment:
    """
    Class representing a real-time fMRI experiment.

    This class handles the setup of the experiment, initialization of the preprocessing
    pipeline, and the management of incoming data from the scanner.

    Parameters
    ----------
    options : Options
        Configuration object containing experiment parameters (e.g., TR, number of volumes, paths).
    sync.hit : multiprocessing.Event
        Event used to signal a CAP hit.
    sync.end : multiprocessing.Event
        Event used to signal the end of the experiment.
    sync.qa_end : multiprocessing.Event
        Event used to signal the end of a QA block.
    """
    def __init__(self, options, sync):
        self.log = set_logger(options.debug, options.silent)

        self.sync = sync

        self.exp_type = options.exp_type

        self.n = 0 # Counter for number of volumes pre-processed (Start = 1)
        self.t = -1 # Counter for number of received volumes (Start = 0
        self.Nv= None # Number of voxels in data mask
        self.Nt = options.nvols # Number acquisitions
        self.TR = options.tr # TR [seconds]

        self.nvols_discard = options.discard # Number of volumes to discard from any analysis (won't enter pre-processing)

        self.this_motion = None

        if options.mask_path is None:
            self.log.error('  Experiment_init_ - No mask was provided!')
            sys.exit(-1)
        else:
            self.mask_img  = load_fMRI_file(options.mask_path)
            self.mask_Nv = int(np.sum(self.mask_img.get_fdata()))
            self.log.debug(f'  Experiment_init_ - Number of Voxels in user-provided mask: {self.mask_Nv}')

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

        # Keep a record of motion estimates
        self.this_motion = [i[self.t] for i in motion]
        self.pipe.motion_estimates.append(self.this_motion)

        if len(self.this_motion) != 6:
            self.log.error('Motion not read in correctly.')
            self.log.error(f'Expected length: 6 | Actual length: {len(self.this_motion)}')
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
        return self.pipe.process(self.t, self.n, self.this_motion, this_t_data)

    def compute_TR_data(self, motion, extra):
        self._compute_TR_data_impl(motion, extra)
        self.log.info(f' - Time point [t={self.t}, n={self.n}]')
        return 1
    
    def end_run(self, save=True):
        """Finalize the experiment by saving all outputs and signaling completion."""
        if save:
            self.pipe.final_steps()
        self.sync.end.set()


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
        Object that performs spatial pattern matching with templates.

    hits : np.ndarray
        2D array tracking detected hits [template x time].

    hit_detector : HitDetector
        Object that decides if a hit has occured.
    """
    def __init__(self, options, sync):
        super().__init__(options, sync)

        self.lastQA_endTR  = 0
        self.out_dir = options.out_dir
        self.out_prefix = options.out_prefix

        self.qa_onsets = []
        self.qa_offsets = []
        self.qa_onsets_path  = osp.join(self.out_dir,self.out_prefix+'.qa_onsets.txt')
        self.qa_offsets_path = osp.join(self.out_dir,self.out_prefix+'.qa_offsets.txt')
        
        matching_opts = MatchingOpts(**options.matching)
        hit_opts = HitOpts(**options.hits)

        self.hit_thr = options.hit_thr

        self.vols_noqa = matching_opts.vols_noqa
        self.match_method = matching_opts.match_method
        self.match_start = matching_opts.match_start
        
        base_arr = np.zeros((self.mask_Nv, self.Nt), dtype=np.float32)
        self.shm_tr = SharedMemory(create=True, size=base_arr.nbytes, name="tr_data")
        self.shared_tr_data = np.ndarray(base_arr.shape, dtype=base_arr.dtype, buffer=self.shm_tr.buf)
        self.shared_tr_index = Value(c_int, -1)
        
        manager = Manager()
        self.shared_qa_onsets = manager.list()
        self.shared_qa_offsets = manager.list()

        try:
            matcher_cls = Matcher.from_name(matching_opts.match_method)
            self.matcher = matcher_cls(matching_opts, options.match_path, self.Nt, self.sync.end, sync.new_tr, sync.shm_ready)
        except ValueError as e:
            self.log.error(f"Matcher setup failed: {e}")
            sync.end.set()
            sys.exit(-1)

        self.hits = np.zeros((self.matcher.Ntemplates, self.Nt))
        self.hit_detector = HitDetector(hit_opts, self.hit_thr)
        self.last_hit = None
        
        self.matching_opts = matching_opts
        
    def compute_TR_data(self, motion, extra):
        hit_status = self.sync.hit.is_set()
        qa_end_status = self.sync.qa_end.is_set()
        
        self.sync.tr_index.value = self.t

        processed = super()._compute_TR_data_impl(motion, extra)

        if qa_end_status:
            self.lastQA_endTR = self.t
            self.qa_offsets.append(self.t)
            self.shared_qa_offsets.append(self.t)
            self.sync.qa_end.clear()
            self.log.info(f'QA ended (cleared) --> updating lastQA_endTR = {self.lastQA_endTR}')
        
        hit = None
        template_labels = self.matcher.template_labels
        
        in_matching_window = self.t >= self.match_start # Ready to match
        cooldown = self.t < self.lastQA_endTR + self.vols_noqa # Cooldown after a hit

        if self.t == max(0, self.match_start - 1):
            self.hit_detector.calculate_enorm_diff(self.this_motion) # Feed in motion before matching begins

        if in_matching_window: 
            scores = self.matcher.match(self.t, self.n, processed)

            info_text = f' - Time point [t={self.t}, n={self.n}]'
            if self.last_hit:
                info_text += f' | Last hit: {self.last_hit}'
            self.log.info(info_text)

            if not (hit_status or cooldown): # If waiting for hit
                hit = self.hit_detector.detect(self.t, template_labels, scores, self.this_motion)
        
                if hit:
                    self.sync.hit.clear()
                    self.log.info(f'[t={self.t},n={self.n}] ==========================================  Template hit [{hit}]')
                    self.qa_onsets.append(self.t)
                    self.shared_qa_onsets.append(self.t)
                    self.hits[template_labels.index(hit), self.t] = 1
                    self.shared_tr_data[:, self.t] = processed.ravel()
                    self.sync.hit.set()
                    self.last_hit = hit

        else:
            self.log.info(f' - Time point [t={self.t}, n={self.n}] | Matching begins at t={self.match_start}')

        return 1

    def start_streaming(self, shared_responses):
        streamer_config = StreamingConfig(
            self.Nt,
            self.matcher.template_labels,
            self.hit_thr,
            self.matching_opts,
            self.mask_img,
            self.mask_Nv,
            self.out_dir,
            self.out_prefix
        )

        self.mp_prc_stream = Process(target=run_streamer, args=(
            streamer_config,
            self.sync,
            self.shared_qa_onsets,
            self.shared_qa_offsets,
            shared_responses
        ))

        self.mp_prc_stream.start()

    def get_enabled_step_config(self, step_name):
        for step_cfg in self.pipe.step_opts:
            if step_cfg.get("name", "").lower() == step_name.lower() and step_cfg.get("enabled", False):
                return step_cfg
        return None

    def write_hit_arrays(self):
        """Save match scores and hit arrays"""
        match_scores_path = osp.join(self.out_dir,self.out_prefix+f'.{self.match_method}_scores')
        np.save(match_scores_path, self.matcher.scores)
        self.log.info(f"Saved match scores to {match_scores_path + '.npy'}")

        hits_path = osp.join(self.out_dir,self.out_prefix+'.hits')
        np.save(hits_path, self.hits)
        self.log.info('Saved hits info to %s' % hits_path)
    
    def write_hit_maps(self):
        """Write out the maps associated with the hits"""
        if self.exp_type == "esam" or self.exp_type == "esam_test":
            Hits_DF = pd.DataFrame(self.hits.T, columns=self.matcher.template_labels)
            for template in self.matcher.template_labels:
                this_template_hits = Hits_DF[template].sum()
                if this_template_hits > 0: # There were hits for this particular template
                    hit_ID = 1
                    for vol in Hits_DF[Hits_DF[template]==True].index:
                        if (cfg := self.get_enabled_step_config(StepType.WINDOWING.value)): # If windowing is enabled
                            win_length = cfg.get("win_length", 4) # win_length is 4 by default if not present in config
                            this_template_vols = vol-np.arange(win_length+self.hit_detector.nconsec_vols-1)
                        else:
                            this_template_vols = vol-np.arange(self.hit_detector.nconsec_vols)
                        out_file = osp.join(self.out_dir, self.out_prefix + '.Hit_'+template+'_'+str(hit_ID).zfill(2)+'.nii')
                        self.log.info(' - final_steps - [%s-%d]. Contributing Vols: %s | File: %s' % (template, hit_ID,str(this_template_vols), out_file ))
                        this_template_InMask  = self.pipe.Data_processed[:,this_template_vols].mean(axis=1)
                        self.log.debug(' - final_steps - this_template_InMask.shape %s' % str(this_template_InMask.shape))
                        unmask_fMRI_img(this_template_InMask, self.mask_img, out_file)
                        hit_ID += 1
    
    def write_qa(self):
        """Write out QA_Onsets and QA_Offsets"""
        with open(self.qa_onsets_path,'w') as file:
            for onset in self.qa_onsets:
                file.write("%i\n" % onset)
        with open(self.qa_offsets_path,'w') as file:
            for offset in self.qa_offsets:
                file.write("%i\n" % offset)

    def end_run(self, save=True):
        """Finalize the experiment by saving all outputs and signaling completion."""
        if save:
            self.pipe.final_steps()

        # TODO: move hit saving to HitDetector?
        self.write_hit_arrays()
        self.write_hit_maps()
        self.write_qa()
        self.shm_tr.close()
        self.shm_tr.unlink()

        self.sync.end.set()