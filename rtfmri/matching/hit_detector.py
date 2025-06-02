import numpy as np

from utils.svr_methods import is_hit_rt01

class HitDetector:
    """Class for deciding if a TR counts as a hit based on Matcher's scores"""
    def __init__(self, hit_opts):
        self.hit_method = hit_opts.hit_method
        self.zscore_thr = hit_opts.zscore_thr
        self.nconsec_vols  = hit_opts.nconsec_vols

        # # These don't do anything yet...
        # self.hit_domot = hit_opts.hit_domot
        # self.hit_mot_th = hit_opts.svr_mot_th

        self.hit_method_func = None
        # TODO: stop using string literals and add checks if method doesnt exist
        if self.hit_method == "method01":
            self.hit_method_func = is_hit_rt01
        else:
            raise ValueError(f"Unknown hit method: {self.hit_method}")
    
    def detect(self, t, template_labels, scores):
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

        Returns
        -------
        str or None
            The label of the template that qualifies as a hit at time `t`, or None if no hit is found.
        """
        return self.hit_method_func(
            t,
            template_labels,
            scores,
            self.zscore_thr,
            self.nconsec_vols
        )
    