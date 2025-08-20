from math import sqrt
from rtcog.matching.hit_utils import is_hit_rt01

from rtcog.utils.log import get_logger
from rtcog.matching.hit_opts import HitOpts

log = get_logger()

class HitDetector:
    """Class for deciding if a TR counts as a hit based on Matcher's scores"""
    def __init__(self, hit_opts: HitOpts):
        self.hit_method = hit_opts.hit_method
        self.zscore_thr = hit_opts.hit_thr
        self.nconsec_vols  = hit_opts.nconsec_vols

        self.do_mot = hit_opts.do_mot
        self.mot_thr = hit_opts.mot_thr

        self.hit_method_func = None
        # TODO: stop using string literals
        if self.hit_method == "method01":
            self.hit_method_func = is_hit_rt01
        else:
            raise ValueError(f"Unknown hit method: {self.hit_method}")

        self.enorm_prev = None
    
    def calculate_enorm_diff(self, motion):
        """Calculate difference in euclidean norm between this and previous TR. Used for motion thresholding."""
        if not self.enorm_prev: # TR 0
            self.enorm_prev = sqrt(sum(x**2 for x in motion))
            return

        enorm_current = sqrt(sum(x**2 for x in motion))
        enorm_diff = enorm_current - self.enorm_prev
        self.enorm_prev = enorm_current
        return enorm_diff
        
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

        return self.hit_method_func(
            t,
            template_labels,
            scores,
            self.zscore_thr,
            self.nconsec_vols
        )
        
    