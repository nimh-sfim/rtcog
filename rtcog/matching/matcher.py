import sys
import pickle
from types import SimpleNamespace
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from rtcog.utils.log import get_logger
from rtcog.matching.matching_utils import rt_svrscore_vol, rt_maskscore_vol

log = get_logger()


class Matcher:
    """Base class for matching processed TR data to given templates"""
    
    registry = {} # Holds all available matching classes

    def __init__(self, match_opts, match_path, Nt, mp_evt_end, mp_new_tr, mp_shm_ready):
        self.match_start = match_opts.match_start # First volume to do decoding on
        self.Nt = Nt
        self.scores = None
        self.Ntemplates = None
        
        self.mp_evt_end = mp_evt_end
        self.mp_new_tr = mp_new_tr
        self.mp_shm_ready = mp_shm_ready

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Skip abstract or helper base classes
        if cls.__name__ == "Matcher" or cls.__name__.startswith("_"):
            return
    
        # Strip "Matcher" from the end of class names
        name = cls.__name__
        if name.endswith("Matcher"):
            name = name[:-7]
        name = name.lower()
        
        cls.registry[name] = cls

    @classmethod
    def from_name(cls, name):
        if name not in cls.registry:
            raise ValueError(f'Unknown matching method: {name}')
        return cls.registry[name]

    def match(self, t, n, tr_data):
        """Compute similarity of TR data to templates"""
        if self.scores is None:
            self.scores = np.zeros((self.Ntemplates, self.Nt))

        this_t_scores = self.func(np.squeeze(tr_data), self.input, self.template_labels)
        self.scores[:, t] = this_t_scores.ravel()
        self.shared_arr[:, t] = this_t_scores.ravel()
        self.mp_new_tr.set()
        log.debug(f'[t={t},n={n}] Online - Matching - scores.shape   {self.scores.shape}')
        
        return self.scores
    
    def setup_shared_memory(self):
        """Create shared memory buffer to pass match scores to data streaming process"""
        if self.Ntemplates is None:
            raise RuntimeError("Ntemplates must be set before creating shared memory")
        base_arr = np.zeros((self.Ntemplates, self.Nt), dtype=np.float32)
        self.shm = SharedMemory(create=True, size=base_arr.nbytes, name="match_scores")
        self.shared_arr = np.ndarray(base_arr.shape, dtype=base_arr.dtype, buffer=self.shm.buf)
        

class SVRMatcher(Matcher):
    """Match to templates using pretrained SVR model"""
    def __init__(self, match_opts, match_path, Nt, mp_evt_end, mp_new_tr, mp_shm_ready):
        super().__init__(match_opts, match_path, Nt, mp_evt_end, mp_new_tr, mp_shm_ready)

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
        
        self.setup_shared_memory()
        self.mp_shm_ready.set()
        

class MaskMatcher(Matcher):
    """Match to templates using pretrained Mask Method"""
    def __init__(self, match_opts, match_path, Nt, mp_evt_end, mp_new_tr, mp_shm_ready):
        super().__init__(match_opts, match_path, Nt, mp_evt_end, mp_new_tr, mp_shm_ready)

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

        self.setup_shared_memory()
        self.mp_shm_ready.set()
