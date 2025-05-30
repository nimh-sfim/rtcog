<<<<<<< HEAD
from math import sqrt
from utils.svr_methods import is_hit_rt01

from utils.log import get_logger

log = get_logger()

class HitDetector:
    """Class for deciding if a TR counts as a hit based on Matcher's scores"""
    def __init__(self, hit_opts, hit_thr):
        self.hit_method = hit_opts.hit_method
        self.zscore_thr = hit_thr
        self.nconsec_vols  = hit_opts.nconsec_vols

        # # These don't do anything yet...
        self.do_mot = hit_opts.do_mot
        self.mot_thr = hit_opts.mot_thr

        self.hit_method_func = None
        # TODO: stop using string literals
=======
import numpy as np

from utils.svr_methods import is_hit_rt01

class HitDetector:
    """Class for deciding if a TR counts as a hit based on Matcher's scores"""
    def __init__(self, hit_opts, mp_evt_hit):
        self.mp_evt_hit = mp_evt_hit
        

        self.hit_method = hit_opts.hit_method
        self.zscore_thr = hit_opts.zscore_thr
        self.nconsec_vols  = hit_opts.nconsec_vols

        # # These don't do anything yet...
        # self.hit_domot = hit_opts.hit_domot
        # self.hit_mot_th = hit_opts.svr_mot_th

        self.hit_method_func = None
        # TODO: stop using string literals and add checks if method doesnt exist
>>>>>>> 328c21c (wip: create HitDetector class, refs #23)
        if self.hit_method == "method01":
            self.hit_method_func = is_hit_rt01
        else:
            raise ValueError(f"Unknown hit method: {self.hit_method}")
<<<<<<< HEAD

        self.enorm_prev = None
        self.enorm_diff = None
    
    def calculate_enorm_diff(self, motion):
        """Calculate difference in euclidean norm between this and previous TR. Used for motion thresholding."""
        if not self.enorm_prev: # TR 0
            self.enorm_prev = sqrt(sum(x**2 for x in motion))
            self.enorm_diff = 0
            return

        enorm_current = sqrt(sum(x**2 for x in motion))
        self.enorm_diff = self.enorm_prev - enorm_current
        self.enorm_prev = enorm_current
        
    def detect(self, t, template_labels, scores, motion=None):
=======
    
    def detect(self, t, template_labels, scores):
>>>>>>> 328c21c (wip: create HitDetector class, refs #23)
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

<<<<<<< HEAD
        motion : list of float
            The 6 motion parameters

=======
>>>>>>> 328c21c (wip: create HitDetector class, refs #23)
        Returns
        -------
        str or None
            The label of the template that qualifies as a hit at time `t`, or None if no hit is found.
        """
<<<<<<< HEAD
        if self.do_mot:
            self.calculate_enorm_diff(motion)
            if self.enorm_diff > self.mot_thr:
                log.info(f'[t={t}] Motion exceeds threshold {self.mot_thr}')
                return

=======
>>>>>>> 328c21c (wip: create HitDetector class, refs #23)
        return self.hit_method_func(
            t,
            template_labels,
            scores,
            self.zscore_thr,
            self.nconsec_vols
        )
<<<<<<< HEAD
        
=======
>>>>>>> 328c21c (wip: create HitDetector class, refs #23)
    