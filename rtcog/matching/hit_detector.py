import numpy as np
from math import sqrt

from rtcog.utils.log import get_logger
from rtcog.matching.hit_opts import HitOpts

log = get_logger()

class HitDetector:
    """Class for deciding if a TR counts as a hit based on Matcher's scores"""
    def __init__(self, hit_opts: HitOpts):
        self.hit_thr = hit_opts.hit_thr
        self.nconsec_vols  = hit_opts.nconsec_vols
        self.nonline  = hit_opts.nonline

        self.do_mot = hit_opts.do_mot
        self.mot_thr = hit_opts.mot_thr

        self.enorm_prev = None
    
    def calculate_enorm_diff(self, motion):
        """Calculate difference in euclidean norm between this and previous TR. Used for motion thresholding."""
        if self.enorm_prev is None: # TR 0
            self.enorm_prev = sqrt(sum(x**2 for x in motion))
            return

        enorm_current = sqrt(sum(x**2 for x in motion))
        enorm_diff = enorm_current - self.enorm_prev
        self.enorm_prev = enorm_current
        return enorm_diff
        
    def is_hit(self, t, template_labels, scores):
        """
        Determines if a specific time point `t` represents a "hit" for a template based on if match scores
        exceed a threshold.

        A time point is a hit if:
        - No more than `nonline` templates are >= `hit_thr` at time `t`.
          - The one with the highest score is selected as the "hit".
        - That same template has also been above `hit_thr` nconsec_vols (including the current volume)
        
        Parameters
        -----------
        t : int
            The current time point.
        template_labels : list of str
            List of template labels corresponding to the rows of `scores`.
        scores : np.ndarray of shape (n_templates, n_timepoints)
            Match scores for each template across time thus far.

        Returns
        -------
        str or None
            The label of the template that qualifies as a hit at time `t`, or None if no hit is found.
        """
        log.debug(f' ============== entering is_hit [{t =}] [{self.hit_thr =}]=======================')
        this_t_scores = scores[:,t]
        this_t_matches = np.where(this_t_scores >= self.hit_thr)[0]
        nmatches = len(this_t_matches)

        log.debug(' === is_hit - this_t_scores: ' + ', '.join(f'{f:.3f}' for f in this_t_scores))
        log.debug(' === is_hit - nmatches %d' % nmatches)
        
        if nmatches == 0 or nmatches > self.nonline:
            return None # Hit is none if more than nonline templates is above threshold, or all are below
        
        this_t_hit = this_t_matches[np.argmax(this_t_scores[this_t_matches])]
        this_t_hit_label = template_labels[this_t_hit]

        start = t - self.nconsec_vols + 1
        if start < 0:
            return None

        # Ensure template exceeds threshold for nconsec_vols time points (including current)
        prev_match = scores[:, t - self.nconsec_vols + 1 : t]
        if np.all(prev_match[this_t_hit] >= self.hit_thr):
            return this_t_hit_label

        return None

    
    def detect(self, t, template_labels, scores, motion=None):
        """
        Detect if a hit occured, and if so, return the name of the template.
        
        Parameters
        ----------
        t : int
            The current TR.
        template_labels : list of str
            List of template labels.
        scores : np.ndarray
            The scores for each template from Matcher.
        motion : list of float
            The 6 motion parameters

        Returns
        -------
        str or None
            The label of the template that qualifies as a hit at time `t`, or None if no hit is found.
        """
        if self.do_mot:
            enorm_diff = self.calculate_enorm_diff(motion)
            if enorm_diff > self.mot_thr:
                log.info(f'[t={t}] Motion exceeds threshold {self.mot_thr}')
                return

        return self.is_hit(
            t,
            template_labels,
            scores,
        )
        
    