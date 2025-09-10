import sys
import os.path as osp
from multiprocessing import Process
from multiprocessing import Value, Manager
from multiprocessing.shared_memory import SharedMemory
from ctypes import c_int

import numpy as np
import pandas as pd

from rtcog.processor.preproc_processor import PreprocProcessor
from rtcog.preproc.step_types import StepType
from rtcog.matching.matcher import Matcher
from rtcog.matching.matching_opts import MatchingOpts
from rtcog.matching.hit_opts import HitOpts
from rtcog.matching.hit_detector import HitDetector
from rtcog.viz.esam_streaming import run_streamer
from rtcog.viz.streaming_config import StreamingConfig
from rtcog.utils.fMRI import unmask_fMRI_img
from rtcog.utils.sync import QAState
from rtcog.viz.score_plotter import ScorePlotter


class ESAMProcessor(PreprocProcessor):
    """
    Real-time fMRI processor class supporting experience sampling (ESAM) mode.

    Extends `Processor` with template matching, hit detection, QA state tracking,
    and dynamic Panel-based GUI streaming.

    Parameters
    ----------
    options : Options
        Experiment configuration options.
    sync : SyncEvents
        Multiprocessing event signals for synchronization.
    minimal : bool, optional
        If True, generate a static score report at the end.

    Attributes
    ----------
    lastaction_endTR : int
        Most recent QA offset TR.
    matcher : Matcher
        Template matcher instance.
    shared_tr_data : np.ndarray
        Shared memory array for matched volumes.
    shared_qa_onsets : ListProxy
        Shared list of QA onsets.
    shared_qa_offsets : ListProxy
        Shared list of QA offsets.
    hits : np.ndarray
        Hit detection matrix [template x TR].
    """
    def __init__(self, options, sync, minimal=False):
        super().__init__(options, sync)

        self.lastaction_endTR  = 0
        self.out_dir = options.out_dir
        self.out_prefix = options.out_prefix

        self.qa_onsets = []
        self.qa_offsets = []
        self.qa_onsets_path  = osp.join(self.out_dir,self.out_prefix+'.qa_onsets.txt')
        self.qa_offsets_path = osp.join(self.out_dir,self.out_prefix+'.qa_offsets.txt')
        
        self.match_opts = MatchingOpts(**options.matching)
        self.hit_opts = HitOpts(**options.hits, hit_thr=options.hit_thr)

        base_arr = np.zeros((self.mask_Nv, self.Nt), dtype=np.float32)
        self.shm_tr = SharedMemory(create=True, size=base_arr.nbytes, name="tr_data")
        self.shared_tr_data = np.ndarray(base_arr.shape, dtype=base_arr.dtype, buffer=self.shm_tr.buf)
        self.shared_tr_index = Value(c_int, -1)
        
        manager = Manager()
        self.shared_qa_onsets = manager.list()
        self.shared_qa_offsets = manager.list()

        try:
            matcher_cls = Matcher.from_name(self.match_opts.match_method)
            self.matcher = matcher_cls(self.match_opts, options.match_path, self.Nt, self.sync.end, sync.new_tr, sync.shm_ready)
        except ValueError as e:
            self.log.error(f"Matcher setup failed: {e}")
            sync.end.set()
            sys.exit(-1)

        self.hits = np.zeros((self.matcher.Ntemplates, self.Nt))
        self.hit_detector = HitDetector(self.hit_opts)
        self.last_hit = None
        
        self.minimal = minimal
        
        self.streaming_config = StreamingConfig(
            self.Nt,
            self.matcher.template_labels,
            self.hit_opts.hit_thr,
            self.match_opts,
            self.mask_img,
            self.mask_Nv,
            self.out_dir,
            self.out_prefix
        )
        
        if self.match_opts.match_start >= self.Nt:
            self.log.warning(
                f"Matching will never occur: match_start ({self.match_opts.match_start}) "
                f"is >= total volumes ({self.Nt})."
            )
        
    def compute_TR_data(self, motion, extra):
        """
        Process one TR in ESAM mode with template matching and hit detection.

        Parameters
        ----------
        motion : list of list[float]
            6 motion parameters per TR.
        extra : list of list[float]
            Voxel-wise time series.

        Returns
        -------
        int
            Always returns 1.
        """
        hit_status = self.sync.hit.is_set()
        action_end_status = self.sync.action_end.is_set()
        
        self.sync.tr_index.value = self.t

        processed = super()._compute_TR_data_impl(motion, extra)

        if action_end_status:
            self.lastaction_endTR = self.t
            self.qa_offsets.append(self.t)
            self.shared_qa_offsets.append(self.t)
            self.sync.action_end.clear()
            self.log.info(f'QA ended (cleared) --> updating lastaction_endTR = {self.lastaction_endTR}')
        
        hit = None
        template_labels = self.matcher.template_labels
        
        in_matching_window = self.t >= self.match_opts.match_start # Ready to match
        cooldown = self.t < self.lastaction_endTR + self.match_opts.vols_noqa # Cooldown after a hit

        if self.t == max(0, self.match_opts.match_start - 1):
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
            self.log.info(f' - Time point [t={self.t}, n={self.n}] | Matching begins at t={self.match_opts.match_start}')

        return 1

    def start_streaming(self, shared_responses):
        """
        Launch background Panel streaming server for real-time visualization.

        Parameters
        ----------
        shared_responses : DictProxy
            Shared-memory dictionary for real-time participant responses.
        """
        self.mp_prc_stream = Process(target=run_streamer, args=(
            self.streaming_config,
            self.sync,
            self.shared_qa_onsets,
            self.shared_qa_offsets,
            shared_responses
        ))

        self.mp_prc_stream.start()

    def get_enabled_step_config(self, step_name):
        """
        Return config dict for enabled pipeline step by name.

        Parameters
        ----------
        step_name : str
            Name of the step (e.g., "windowing").

        Returns
        -------
        dict or None
            Step config dictionary if enabled, otherwise None.
        """
        for step_cfg in self.pipe.step_opts:
            if step_cfg.get("name", "").lower() == step_name.lower() and step_cfg.get("enabled", False):
                return step_cfg
        return None

    def write_hit_arrays(self):
        """Save match scores and hit arrays"""
        match_scores_path = osp.join(self.out_dir,self.out_prefix+f'.{self.match_opts.match_method}_scores')
        np.save(match_scores_path, self.matcher.scores)
        self.log.info(f"Saved match scores to {match_scores_path + '.npy'}")

        hits_path = osp.join(self.out_dir,self.out_prefix+'.hits')
        np.save(hits_path, self.hits)
        self.log.info('Saved hits info to %s' % hits_path)
    
    def write_hit_maps(self):
        """Write out the maps associated with the hits"""
        Hits_DF = pd.DataFrame(self.hits.T, columns=self.matcher.template_labels)
        for template in self.matcher.template_labels:
            this_template_hits = Hits_DF[template].sum()
            if this_template_hits > 0: # There were hits for this particular template
                hit_ID = 1
                for vol in Hits_DF[Hits_DF[template]==True].index:
                    if (cfg := self.get_enabled_step_config(StepType.WINDOWING.value)): # If windowing is enabled
                        win_length = cfg.get("win_length", 4) # win_length is 4 by default if not present in config
                        this_template_vols = vol-np.arange(win_length+self.hit_opts.nconsec_vols-1)
                    else:
                        this_template_vols = vol-np.arange(self.hit_opts.nconsec_vols)
                    out_file = osp.join(self.out_dir, self.out_prefix + '.Hit_'+template+'_'+str(hit_ID).zfill(2)+'.nii')
                    self.log.info(' - final_steps - [%s-%d]. Contributing Vols: %s | File: %s' % (template, hit_ID,str(this_template_vols), out_file ))
                    this_template_InMask  = self.pipe.Data_processed[:,this_template_vols].mean(axis=1)
                    self.log.debug(' - final_steps - this_template_InMask.shape %s' % str(this_template_InMask.shape))
                    unmask_fMRI_img(this_template_InMask, self.mask_img, out_file)
                    hit_ID += 1
    
    def write_qa(self):
        """Save QA onsets and offsets to plain-text files."""
        with open(self.qa_onsets_path,'w') as file:
            for onset in self.qa_onsets:
                file.write("%i\n" % onset)
        with open(self.qa_offsets_path,'w') as file:
            for offset in self.qa_offsets:
                file.write("%i\n" % offset)

    def write_report(self):
        if self.matcher.scores is None:
            self.log.warning("No matching scores were computed; skipping report generation.")
            return

        match_scores_df = pd.DataFrame(
        self.matcher.scores.T,
        columns=self.matcher.template_labels
    )
        qa_state = QAState(
            self.qa_onsets,
            self.qa_offsets,
            False,
            False,
            False,
            None
        )

        plotter = ScorePlotter(self.streaming_config, streaming=False)
        plotter.render_static(match_scores_df, qa_state) 
        
    def end_run(self, save=True):
        """
        Finalize the experiment, including file output and memory cleanup.

        Parameters
        ----------
        save : bool
            Whether to save final output files (default: True).
        """
        if save:
            self.pipe.final_steps()

        # TODO: move hit saving to HitDetector?
        self.write_hit_arrays()
        self.write_hit_maps()
        self.write_qa()
        if self.minimal:
            self.write_report()
        self.shm_tr.close()
        self.shm_tr.unlink()
        self.matcher.shm.close()
        self.matcher.shm.unlink()

        self.sync.end.set()