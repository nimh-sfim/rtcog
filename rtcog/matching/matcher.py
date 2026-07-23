import sys
import pickle
import numpy as np

from rtcog.matching.matching_opts import MatchingOpts
from rtcog.matching.matching_utils import (
    nmi_bin_data,
    nmi_from_bins,
    nmi_n_bins,
    pearson_correlations,
)
from rtcog.utils.log import get_logger
from rtcog.utils.shared_memory_manager import SharedMemoryManager
from rtcog.utils.sync import SyncEvents

log = get_logger()


# TODO: big refactor of on/offline matching (see docs/matching_architecture_plan.md)
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
            self.mp_end.set()
            raise ValueError('SVR Model not provided.')
        
        try:
            with open(match_path, "rb") as f:
                self.input = pickle.load(f)
        except Exception as e:
            self.mp_end.set()
            raise RuntimeError(f'Unable to open SVR model pickle file: {e}')

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
            self.mp_end.set()
            raise ValueError('Template info for match method not provided.')
        
        try:
            self.input = np.load(match_path, allow_pickle=True)
        except Exception as e:
            self.mp_end.set()
            raise RuntimeError(f'Error loading mask method file: {e}')
        
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
        

class PearsonMatcher(Matcher):
    """Match TRs to template maps using spatial Pearson correlation."""

    def __init__(self, match_opts, Nt, sync, match_path):
        super().__init__(match_opts, Nt, sync, match_path)

        if match_path is None:
            self.mp_end.set()
            raise ValueError('Pearson template data not provided.')

        try:
            self.input = np.load(match_path, allow_pickle=True)
        except Exception as e:
            self.mp_end.set()
            raise RuntimeError(f'Error loading Pearson template file: {e}')

        if "labels" not in self.input or "templates" not in self.input:
            self.mp_end.set()
            raise ValueError('Pearson template file must contain "labels" and "templates".')

        self.template_labels = list(self.input["labels"])
        self.Ntemplates = len(self.template_labels)
        templates = np.asarray(self.input["templates"], dtype=np.float32)
        if templates.ndim != 2 or templates.shape[0] != self.Ntemplates:
            self.mp_end.set()
            raise ValueError(
                f'Pearson templates must have shape (n_templates, n_voxels); '
                f'got {templates.shape} for {self.Ntemplates} labels.'
            )

        self.Nvoxels = templates.shape[1]
        self.template_centered = templates - templates.mean(axis=1, keepdims=True)
        self.template_norms = np.linalg.norm(self.template_centered, axis=1)

        log.info(f'List of templates to be tested: {self.template_labels}')

        self.setup_shared_memory()
        self.mp_shm_ready.set()

    def _match(self, tr_data):
        """Return the spatial Pearson correlation with each template."""
        data = np.squeeze(tr_data).astype(np.float32).ravel()
        if data.size != self.Nvoxels:
            raise ValueError(
                f'Pearson matcher expected {self.Nvoxels} voxels, got {data.size}'
            )

        return pearson_correlations(
            data,
            self.template_centered,
            self.template_norms,
        )


class NMIMatcher(Matcher):
    """
    Match TRs to templates using signed normalized mutual information.

    This matcher scores TRs with binned normalized mutual information inspired
    by gRAICAR (https://github.com/yangzhi-psy/gRAICAR). We also include the raw
    templates so Pearson correlation can supply the score sign. High NMI with
    negative correlation is reported as a negative score.

    Parameters
    ----------
    match_opts : MatchingOpts
        Runtime matching options.
    Nt : int
        Total number of time points.
    sync : SyncEvents
        Synchronization events shared with the processor.
    match_path : str
        Path to an ``.npz`` file containing ``labels`` and raw ``templates``.
        Templates must have shape ``(n_templates, n_voxels)``. ``template_bins``
        and ``n_bins`` may also be supplied as precomputed cache values.
    """
    def __init__(self, match_opts, Nt, sync, match_path):
        super().__init__(match_opts, Nt, sync, match_path)

        if match_path is None:
            self.mp_end.set()
            raise ValueError('NMI template data not provided.')

        try:
            self.input = np.load(match_path, allow_pickle=True)
        except Exception as e:
            self.mp_end.set()
            raise RuntimeError(f'Error loading NMI template file: {e}')

        self.template_labels = list(self.input["labels"])
        self.Ntemplates = len(self.template_labels)

        if "templates" not in self.input:
            self.mp_end.set()
            raise ValueError('NMI template file must contain raw "templates" for signed-correlation scoring.')

        templates = np.asarray(self.input["templates"], dtype=np.float32)
        if templates.ndim != 2 or templates.shape[0] != self.Ntemplates:
            self.mp_end.set()
            raise ValueError(
                f'NMI templates must have shape (n_templates, n_voxels); '
                f'got {templates.shape} for {self.Ntemplates} labels.'
            )

        self.templates = templates
        self.Nvoxels = self.templates.shape[1]
        self.n_bins = int(np.asarray(self.input["n_bins"]).item()) if "n_bins" in self.input else nmi_n_bins(self.Nvoxels)

        if "template_bins" in self.input:
            template_bins = np.asarray(self.input["template_bins"])
            if template_bins.shape != self.templates.shape:
                self.mp_end.set()
                raise ValueError(
                    f'NMI template_bins shape {template_bins.shape} does not match '
                    f'templates shape {self.templates.shape}.'
                )
            self.template_bins = template_bins.astype(np.int16)
        else:
            self.template_bins = np.vstack([
                nmi_bin_data(template, self.n_bins) for template in self.templates
            ])

        self.template_centered = self.templates - self.templates.mean(axis=1, keepdims=True)
        self.template_norms = np.linalg.norm(self.template_centered, axis=1)

        log.info(f'List of templates to be tested: {self.template_labels}')

        self.setup_shared_memory()
        self.mp_shm_ready.set()

    def _match(self, tr_data):
        """
        Compute MI scores for one processed TR.
        
        Because NMI can be high for inverted maps, Pearson correlation supplies
        the sign of the binned NMI score.

        Parameters
        ----------
        tr_data : array_like
            Processed TR data vector in mask space.

        Returns
        -------
        np.ndarray
            One score per template. Scores are ``sign(correlation) * (NMI - 1)``.
            Zero or non-finite correlations receive a score of zero.

        Raises
        ------
        ValueError
            If ``tr_data`` does not contain the expected number of voxels.
        """
        data = np.squeeze(tr_data).astype(np.float32).ravel()
        if data.size != self.Nvoxels:
            raise ValueError(
                f'NMI matcher expected {self.Nvoxels} voxels, got {data.size}'
            )

        correlations = pearson_correlations(
            data,
            self.template_centered,
            self.template_norms,
        )

        valid = np.isfinite(correlations) & (correlations != 0)
        if not np.any(valid):
            return np.zeros(self.Ntemplates, dtype=np.float32)

        # Perform gRAICAR-style binned NMI
        data_bins = nmi_bin_data(data, self.n_bins)
        scores = np.zeros(self.Ntemplates, dtype=np.float32)
        for idx in np.flatnonzero(valid):
            score = nmi_from_bins(data_bins, self.template_bins[idx], self.n_bins) - 1
            strength = max(score, 0)
            scores[idx] = strength if correlations[idx] > 0 else -strength

        return scores
