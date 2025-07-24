import numpy as np

from rtfmri.utils.sync import QAState
from rtfmri.viz.streaming_config import StreamingConfig

class Plotter:
    """Base class for plotters"""
    data_key = ""

    def __init__(self, config: StreamingConfig):
        self._Nt = config.Nt
        self._template_labels = config.template_labels
        self._Ntemplates = len(self._template_labels)

    def update(self, t: int, data: np.ndarray, qa_state: QAState) -> None:
        """Update the plot with new data"""
        raise NotImplementedError

    def close(self):
        """Optional: What to do at the end of the experiment"""
        pass