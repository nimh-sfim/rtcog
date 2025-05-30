import sys
import pickle
import numpy as np

sys.path.append('..')
from utils.log import get_logger
from utils.core import create_win
from utils.rt_functions import rt_svrscore_vol

log = get_logger()

class Matcher:
    """Base class for matching processed TR data to given templates"""
    def __init__(self, options):
        self.dec_start_vol = options.dec_start_vol # First volume to do decoding on
        self.scores = None
    
    def match(self, t, n, tr_data):
        raise NotImplementedError

class SVRMatcher(Matcher):
    """Match to templates using pretrained SVR model"""
    def __init__(self, options):
        super().__init__(options)

        if options.svr_path is None:
            self.log.error('SVR Model not provided. Program will exit.')
            self.mp_evt_end.set()
            sys.exit(-1)
        self.svr_path = options.svr_path
        
        try:
            with open(self.svr_path, "rb") as f:
                self.SVRs = pickle.load(f)
        except Exception as e:
            log.error('Unable to open SVR model file.')
            log.error(e)
            self.mp_evt_end.set()
            sys.exit(-1)

        self.Ntemplates = len(self.SVRs.keys())
        self.templates_labels = list(self.SVRs.keys())
        log.info(f'List of templates to be tested: {self.templates_labels}')
        
        self.scores = np.zeros((self.Ntemplates, 1))

        self.hit_dowin = options.hit_dowin

        self.hit_dowin = options.hit_dowin
        self.hit_wl = options.hit_wl
        
        if self.hit_dowin:
            self.hit_win_weights = create_win(self.hit_wl)


    def match(self, t, n, tr_data):
        if self.hit_dowin:
            tr_data = self._do_windowing(t, n, tr_data)
        
        if t < self.dec_start_vol: # Wait until after iGLM is stable to perform matching
            self.scores = np.append(self.scores, np.zeros((self.Ntemplates,1)), axis=1)
        else:
            this_t_scores = rt_svrscore_vol(np.squeeze(tr_data), self.SVRs, self.templates_labels)
            self.scores   = np.append(self.scores, this_t_scores, axis=1)
        log.debug(f'[t={t},n={n}] Online - SVRs - scores.shape   {self.scores.shape}')
        
        return self.scores
    
    def _do_windowing(self, t, n, tr_data): 
        if (t >= self.dec_start_vol):
            vols_in_win       = (np.arange(t-4,t)+1)[::-1]
            out_data_windowed = np.dot(tr_data[:,vols_in_win], self.hit_win_weights)
        log.debug(f'[t={t},n={n}] Online - SVRs - out_data_windowed.shape   {out_data_windowed.shape}')
        return out_data_windowed
