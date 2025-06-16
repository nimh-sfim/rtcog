import sys
import os.path as osp
from types import SimpleNamespace

import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas

from preproc.pipeline import Pipeline
from matching.matcher import SVRMatcher, MaskMatcher
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
        self.log.info(f' - Time point [t={self.t}, n={self.n}]')
        return 1
    
    def end_run(self, save=True):
        """Finalize the experiment by saving all outputs and signaling completion."""
        if save:
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
        Object that performs spatial pattern matching with templates.

    hits : np.ndarray
        2D array tracking detected hits [template x time].

    hit_detector : HitDetector
        Object that decides if a hit has occured.
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

        self.hit_thr = options.hit_thr

        self.vols_noqa = matching_opts.vols_noqa
        self.match_method = matching_opts.match_method
        self.win_length = matching_opts.win_length
        
        # TODO: have cleaner way of instantiating these. If new matching methods are added, this would get annoying
        if self.match_method == "svr":
            self.matcher = SVRMatcher(matching_opts, options.match_path, self.Nt, self.mp_evt_end)
        elif self.match_method == 'mask_method':
            self.matcher = MaskMatcher(matching_opts, options.match_path, self.Nt, self.mp_evt_end)

        self.hits = np.zeros((self.matcher.Ntemplates, self.Nt))
        self.hit_detector = HitDetector(hit_opts, self.hit_thr)
        self.last_hit = None

        self.outhtml = osp.join(self.out_dir, self.out_prefix+'.dyn_report')
        
        
    def compute_TR_data(self, motion, extra):
        # TODO: see why I'm getting off by one for hits and the numbers are slightly different.
        hit_status    = self.mp_evt_hit.is_set()
        qa_end_status = self.mp_evt_qa_end.is_set()

        processed = super()._compute_TR_data_impl(motion, extra)
        if self.t > self.nvols_discard - 1:
            scores = self.matcher.match(self.t, self.n, processed)

        if qa_end_status:
            self.lastQA_endTR = self.t
            self.qa_offsets.append(self.t)
            self.mp_evt_qa_end.clear()
            self.log.info(f'QA ended (cleared) --> updating lastQA_endTR = {self.lastQA_endTR}')
        
        template_labels = self.matcher.template_labels
        if hit_status or (self.t <= self.lastQA_endTR + self.vols_noqa):
            info_text = f' - Time point [t={self.t}, n={self.n}]'
            if self.last_hit:
                info_text += f' | Last hit: {self.last_hit}'
            self.log.info(info_text)
            hit = None
        else:
            hit = self.hit_detector.detect(self.t, template_labels, scores)
        
        if hit:
            self.log.info(f'[t={self.t},n={self.n}] =============================================  CAP hit [{hit}]')
            self.qa_onsets.append(self.t)
            self.hits[template_labels.index(hit), self.t] = 1
            self.mp_evt_hit.set()
            self.last_hit = hit
            
        return 1

    def write_hit_arrays(self):
        """Save match scores and hit arrays"""
        match_scores_path = osp.join(self.out_dir,self.out_prefix+f'.{self.match_method}_scores')
        np.save(match_scores_path, self.matcher.scores)
        self.log.info(f"Saved match scores to {match_scores_path + '.npy'}")

        hits_path = osp.join(self.out_dir,self.out_prefix+'.hits')
        np.save(hits_path, self.hits)
        self.log.info('Saved hits info to %s' % hits_path)
    
    def write_dynamic_report(self):
        """Save html file with interactive plot of match scores and hits"""
        match_Scores_DF = pd.DataFrame(self.matcher.scores.T, columns=self.matcher.template_labels)
        match_Scores_DF['TR'] = match_Scores_DF.index
        match_scores_curve     = match_Scores_DF.hvplot(legend='top', label='match Scores', x='TR').opts(width=1500)
        Threshold_line      = hv.HLine(self.hit_thr).opts(color='black', line_dash='dashed', line_width=1)
        Hits_ToPlot = self.hits.T * self.matcher.scores.T
        Hits_ToPlot[Hits_ToPlot==0.0] = None
        Hits_DF = pd.DataFrame(Hits_ToPlot, columns=self.matcher.template_labels)
        Hits_DF['TR'] = Hits_DF.index
        Hits_Marks  = Hits_DF.hvplot(
            legend='top', label='match Scores', 
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
        plot_layout = (match_scores_curve * Threshold_line * Hits_Marks * QA_periods * WAIT_periods).opts(title=f'Experimental Run Results: {self.out_prefix}, {self.match_method}')
        renderer    = hv.renderer('bokeh')
        renderer.save(plot_layout, self.outhtml)
        self.log.info('Dynamic Report written to disk: [%s.html]' % self.outhtml)
        self.log.info('qa_onsets:  %s' % str(self.qa_onsets))
        self.log.info('qa_offsets: %s' % str(self.qa_offsets))
    
    def write_hit_maps(self):
        """Write out the maps associated with the hits"""
        if self.exp_type == "esam" or self.exp_type == "esam_test":
            Hits_DF = pd.DataFrame(self.hits.T, columns=self.matcher.template_labels)
            for template in self.matcher.template_labels:
                this_template_hits = Hits_DF[template].sum()
                if this_template_hits > 0: # There were hits for this particular template
                    hit_ID = 1
                    for vol in Hits_DF[Hits_DF[template]==True].index:
                        if self.matcher.do_win == True:
                            this_template_vols = vol-np.arange(self.win_length+self.hit_detector.nconsec_vols-1)
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

        # TODO: move hit saving to HitDetector
        self.write_hit_arrays()
        self.write_dynamic_report()
        self.write_hit_maps()
        self.write_qa()

        self.mp_evt_end.set()