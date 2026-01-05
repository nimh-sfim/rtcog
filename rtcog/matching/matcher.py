import sys
import pickle
import numpy as np

from rtcog.matching.matching_opts import MatchingOpts
from rtcog.utils.log import get_logger
from rtcog.utils.shared_memory_manager import SharedMemoryManager
from rtcog.utils.sync import SyncEvents

log = get_logger()


# TODO: accept SyncEvents instead of individual mp events
class Matcher:
    """
    Base class for matching processed TR data to given templates.

    This class provides the framework for comparing incoming fMRI volumes against
    predefined brain state templates to detect patterns of interest. Subclasses
    implement specific matching algorithms (e.g., SVR-based or mask-based).

    Attributes
    ----------
    registry : dict
        Class-level registry mapping matcher names to their classes.
    match_start : int
        First volume index to start computing match scores.
    Nt : int
        Total number of time points in the experiment.
    scores : np.ndarray
        Array of match scores, shape (Ntemplates, Nt).
    Ntemplates : int
        Number of templates to match against.
    mp_end : multiprocessing.Event
        Event to signal experiment end.
    mp_new_tr : multiprocessing.Event
        Event set when a new TR is processed.
    mp_shm_ready : multiprocessing.Event
        Event indicating shared memory is ready.

    Methods
    -------
    from_name(name)
        Factory method to instantiate a matcher by name.
    match(t, n, tr_data)
        Compute similarity scores for a TR and update shared memory.
    setup_shared_memory()
        Initialize shared memory for score storage.
    cleanup_shared_memory()
        Clean up shared memory resources.
    _match(tr_data)
        Abstract method for computing match scores (implemented by subclasses).
    """

    registry = {} # Holds all available matching classes

    def __init__(self, match_opts: MatchingOpts, Nt: int, sync: SyncEvents, match_path: str):
        """
        Initialize the Matcher.

        Parameters
        ----------
        match_opts : MatchingOpts
            Configuration options for matching.
        Nt : int
            Total number of time points.
        sync : SyncEvents
            Synchronization events container.
        match_path : str
            Path to matching data (e.g., templates or model).
        """
        self.match_start = match_opts.match_start # First volume to start computing match scores on
        self.Nt = Nt
        self.scores = None
        self.Ntemplates = None
        
        self.mp_end = sync.end
        self.mp_new_tr = sync.new_tr
        self.mp_shm_ready = sync.shm_ready

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
        """
        Compute similarity scores for a TR and update shared memory.

        Parameters
        ----------
        t : int
            Time point index.
        n : int
            Processed volume index.
        tr_data : np.ndarray
            Processed TR data.

        Returns
        -------
        np.ndarray
            Updated scores array.
        """
        if self.scores is None:
            self.scores = np.zeros((self.Ntemplates, self.Nt))
        
        this_t_scores = self._match(tr_data)
        if this_t_scores.ndim != 1:
            raise ValueError(
                f"{self.__class__.__name__}._match() must return 1D array; "
                f"got shape {this_t_scores.shape}"
            )
        if this_t_scores.shape[0] != self.Ntemplates:
            raise ValueError(
                f"{self.__class__.__name__}._match() returned {this_t_scores.shape[0]} "
                f"scores, expected {self.Ntemplates}"
            )

        self.scores[:, t] = this_t_scores
        self.shared_arr[:, t] = this_t_scores
        self.mp_new_tr.set()
        log.debug(f'[t={t},n={n}] Online - Matching - scores.shape   {self.scores.shape}')

        return self.scores

    def setup_shared_memory(self):
        """
        Initialize shared memory for score storage.

        Creates a shared memory buffer to pass match scores to the data streaming process.
        """
        if self.Ntemplates is None:
            raise RuntimeError("Ntemplates must be set before creating shared memory")

        base_arr = np.zeros((self.Ntemplates, self.Nt), dtype=np.float32)
        self.shm_manager = SharedMemoryManager("match_scores", create=True, size=base_arr.nbytes)
        self.shm = self.shm_manager.open()
        self.shared_arr = np.ndarray(base_arr.shape, dtype=base_arr.dtype, buffer=self.shm.buf)
        
    def cleanup_shared_memory(self):
        """
        Clean up shared memory resources.
        """
        if hasattr(self, 'shm_manager'):
            self.shm_manager.cleanup()
        if hasattr(self, 'shm_manager'):
            self.shm_manager.cleanup()
    
    def _match(self, tr_data):
        """
        Abstract method for computing match scores.

        Parameters
        ----------
        tr_data : np.ndarray
            Processed TR data.

        Returns
        -------
        np.ndarray
            1D array of match scores for each template.
        """
        raise NotImplementedError
        

class SVRMatcher(Matcher):
    """
    Match to templates using pretrained SVR model.

    This matcher uses a support vector regression model trained on previous data
    to detect activation patterns in incoming TRs.
    """
    def __init__(self, match_opts, Nt, sync, match_path):
        super().__init__(match_opts, Nt, sync, match_path)

        if match_path is None:
            log.error('SVR Model not provided. Program will exit.')
            self.mp_end.set()
            sys.exit(-1)
        
        try:
            with open(match_path, "rb") as f:
                self.input = pickle.load(f)
        except Exception as e:
            log.error('Unable to open SVR model pickle file.')
            log.error(e)
            sys.exit(-1)

        self.Ntemplates = len(self.input.keys())
        self.template_labels = list(self.input.keys())
        log.info(f'List of templates to be tested: {self.template_labels}')
        
        self.setup_shared_memory()
        self.mp_shm_ready.set()
    
    def _match(self, tr_data):
        out = []
        data = np.squeeze(tr_data)

        for label in self.template_labels:
            out.append(self.input[label].predict(data[:,np.newaxis].T)[0])
        
        return np.array(out)
       

class MaskMatcher(Matcher):
    """Match to templates using pretrained Mask Method"""
    def __init__(self, match_opts, Nt, sync, match_path):
        super().__init__(match_opts, Nt, sync, match_path)

        if match_path is None:
            log.error('Template info for match method not provided. Program will exit.')
            self.mp_end.set()
            sys.exit(-1)

        try:
            self.input = np.load(match_path, allow_pickle=True)
        except Exception as e:
            log.error(f'Error loading mask method file: {e}')
            log.error(e)
            sys.exit(-1)
        
        self.template_labels = list(self.input["labels"])
        self.Ntemplates = len(self.template_labels)
        log.info(f'List of templates to be tested: {self.template_labels}')
        
        self.masked_templates = self.input["masked_templates"].item()
        self.masks = self.input["masks"].item()
        self.voxel_counts = self.input["voxel_counts"].item()

        self.setup_shared_memory()
        self.mp_shm_ready.set()
    
    def _match(self, tr_data):
        out = []

        for name in self.template_labels:
            mask = self.masks[name]
            template = self.masked_templates[name]
            masked_data = np.squeeze(tr_data)[mask]        
            out.append(np.dot(template, masked_data) / self.voxel_counts[name])

        return np.array(out)
        

