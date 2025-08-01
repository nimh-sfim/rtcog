import numpy as np

from rtfmri.utils.sync import QAState
from rtfmri.viz.streaming_config import StreamingConfig

class Plotter:
    """
    Base class for real-time fMRI plotters.

    Subclasses should implement specific plotting behavior using the `update()` method,
    and may optionally override `close()` for cleanup at the end of the experiment.

    Attributes
    ----------
    data_key : str
        Identifier for the type of data this plotter displays.
    """

    data_key = ""

    def __init__(self, config: StreamingConfig):
        """
        Initialize the Plotter with basic configuration settings.

        Parameters
        ----------
        config : StreamingConfig
            Configuration object containing session-level metadata.
        """
        self._Nt = config.Nt
        self._template_labels = config.template_labels
        self._Ntemplates = len(self._template_labels)

    def update(self, t: int, data: np.ndarray, qa_state: QAState) -> None:
        """
        Update the plot with new data for the current time repetition (TR).

        This method must be implemented by subclasses.

        Parameters
        ----------
        t : int
            Current TR.
        data : np.ndarray
            Data to be visualized (format depends on subclass).
        qa_state : QAState
            QA-related metadata for the current TR.
        """
        raise NotImplementedError

    def close(self):
        """
        Optional cleanup at the end of the experiment.

        Subclasses can override this method to release resources or finalize output.
        """
        pass