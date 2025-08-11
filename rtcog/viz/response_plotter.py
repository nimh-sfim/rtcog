import numpy as np
import pandas as pd
import panel as pn
from multiprocessing.managers import DictProxy

from rtcog.utils.sync import QAState
from rtcog.viz.streaming_config import StreamingConfig
from rtcog.viz.plotter import Plotter

class ResponsePlotter(Plotter):
    """
    Display participant responses over time in a dynamic DataFrame panel.

    This class extends the `Plotter` base class to visualize responses collected
    during an fMRI scan.
    """
    data_key = "responses"
    def __init__(self, config: StreamingConfig, responses: DictProxy):
        """
        Initialize the ResponsePlotter with configuration and response tracking.

        Sets up an empty DataFrame with columns corresponding to question names,
        and prepares a Panel DataFrame pane to render it in real time.

        Parameters
        ----------
        config : StreamingConfig
            Configuration object used for streaming.
        responses : DictProxy
            Shared-memory dictionary of responses keyed by question name.
        """
        super().__init__(config)
        self._responses = responses
        question_names = list(responses.keys())

        self._df = pd.DataFrame(columns=question_names)
        self.pane = pn.pane.DataFrame(self._df, sizing_mode="stretch_both", max_width=1000)

        self._last_response_t = None
    
    def should_update(self, t, qa_state):
        """
        Whether or not the plot should be updated.
        
        Parameters
        ----------
        t : int
            Current TR.
        qa_state : QAState
            QA-related metadata for the current TR.
        Returns
        ----------
        bool
            Whether the plot should be updated.
        """
        if not qa_state.qa_offsets:
            return False
        latest_offset = qa_state.qa_offsets[-1]
        return self._last_response_t != latest_offset

    def update(self, t: int, data: np.ndarray, qa_state: QAState) -> None:
        """
        Update the response DataFrame with the latest responses if a new one is available.

        Parameters
        ----------
        t : int
            Current TR.
        data : np.ndarray
            The incoming data at the current TR (not used directly in this method).
        qa_state : QAState
            Object holding QA-related state information.
        """
        latest_offset = qa_state.qa_offsets[-1]
        self._last_response_t = latest_offset
        new_row = {qname: self._responses.get(qname, (None, None))[0] for qname in self._df.columns}
        self._df = pd.concat([self._df, pd.DataFrame([new_row])], ignore_index=True)
        self.pane.object = self._df

                
        
