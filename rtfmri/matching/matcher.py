import sys
import pickle
import numpy as np

sys.path.append('..')
from utils.log import get_logger
from utils.core import create_win
from utils.rt_functions import rt_svrscore_vol, rt_maskscore_vol

log = get_logger()

class Matcher:
    """Base class for matching processed TR data to given templates"""
    def __init__(self, match_opts, match_path):
        self.match_start = match_opts.match_start # First volume to do decoding on
        self.scores = None
        self.Ntemplates = None
        self.func = None

        self.do_win = match_opts.do_win
        self.hit_wl = match_opts.hit_wl

        if self.do_win:
            self.hit_win_weights = create_win(self.hit_wl)
    
    def _do_windowing(self, t, tr_data): 
        # TODO: fix bug (need to save last few TRs in buffer)
        if (t >= self.match_start):
            vols_in_win       = (np.arange(t-4,t)+1)[::-1]
            out_data_windowed = np.dot(tr_data[:,vols_in_win], self.hit_win_weights)
            return out_data_windowed
        return tr_data

    def match(self, t, n, tr_data):
        if self.do_win:
            tr_data = self._do_windowing(t, tr_data)
        
        if t < self.match_start: # Wait until after iGLM is stable to perform matching
            self.scores = np.append(self.scores, np.zeros((self.Ntemplates,1)), axis=1)
        else:
            this_t_scores = self.func(np.squeeze(tr_data), self.input, self.template_labels)
            self.scores   = np.append(self.scores, this_t_scores, axis=1)
        log.debug(f'[t={t},n={n}] Online - Matching - scores.shape   {self.scores.shape}')
        
        return self.scores


class SVRMatcher(Matcher):
    """Match to templates using pretrained SVR model"""
    def __init__(self, match_opts, match_path):
        super().__init__(match_opts, match_path)

        if match_path is None:
            self.log.error('SVR Model not provided. Program will exit.')
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
        
        self.scores = np.zeros((self.Ntemplates, 1))

class MaskMatcher(Matcher):
    """Match to templates using pretrained Mask Method"""
    def __init__(self, match_opts, match_path):
        super().__init__(match_opts, match_path)

        if match_path is None:
            self.log.error('Template info for match method not provided. Program will exit.')
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
        
        self.scores = np.zeros((self.Ntemplates, 1))