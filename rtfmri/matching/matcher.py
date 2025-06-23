import sys
import pickle

import numpy as np

sys.path.append('..')
from rtfmri.utils.log import get_logger
from rtfmri.matching.matching_utils import rt_svrscore_vol, rt_maskscore_vol

log = get_logger()


class Matcher:
    """Base class for matching processed TR data to given templates"""
    def __init__(self, match_opts, match_path, Nt, mp_evt_end):
        self.match_start = match_opts.match_start # First volume to do decoding on
        self.Nt = Nt
        self.scores = None
        self.Ntemplates = None
        
        self.mp_evt_end = mp_evt_end

        self.do_win = match_opts.do_win
        
    def match(self, t, n, tr_data):
        if self.scores is None:
            self.scores = np.zeros((self.Ntemplates, self.Nt))

        this_t_scores = self.func(np.squeeze(tr_data), self.input, self.template_labels)
        self.scores[:, t] = this_t_scores.ravel()
        log.debug(f'[t={t},n={n}] Online - Matching - scores.shape   {self.scores.shape}')
        
        return self.scores

class SVRMatcher(Matcher):
    """Match to templates using pretrained SVR model"""
    def __init__(self, match_opts, match_path, Nt, mp_evt_end):
        super().__init__(match_opts, match_path, Nt, mp_evt_end)

        if match_path is None:
            log.error('SVR Model not provided. Program will exit.')
            self.mp_evt_end.set()
            sys.exit(-1)
        
        try:
            with open(match_path, "rb") as f:
                self.input = pickle.load(f)
        except Exception as e:
            log.error('Unable to open SVR model pickle file.')
            log.error(e)
            sys.exit(-1)

        self.func = rt_svrscore_vol
        
        self.Ntemplates = len(self.input.keys())
        self.template_labels = list(self.input.keys())
        log.info(f'List of templates to be tested: {self.template_labels}')
        

class MaskMatcher(Matcher):
    """Match to templates using pretrained Mask Method"""
    def __init__(self, match_opts, match_path, Nt, mp_evt_end):
        super().__init__(match_opts, match_path, Nt, mp_evt_end)

        if match_path is None:
            log.error('Template info for match method not provided. Program will exit.')
            self.mp_evt_end.set()
            sys.exit(-1)

        try:
            self.input = np.load(match_path, allow_pickle=True)
        except Exception as e:
            log.error(f'Error loading mask method file: {e}')
            log.error(e)
            sys.exit(-1)
        
        self.func = rt_maskscore_vol
        
        self.template_labels = list(self.input["labels"])
        self.Ntemplates = len(self.template_labels)
        log.info(f'List of templates to be tested: {self.template_labels}')
        