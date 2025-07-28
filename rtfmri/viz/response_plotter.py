import numpy as np
import pandas as pd
import panel as pn
from multiprocessing.managers import DictProxy

from rtfmri.utils.sync import QAState
from rtfmri.viz.streaming_config import StreamingConfig
from rtfmri.viz.plotter import Plotter

class ResponsePlotter(Plotter):
    data_key = "responses"
    def __init__(self, config: StreamingConfig, responses: DictProxy):
        super().__init__(config)
        self._responses = responses
        question_names = list(responses.keys())

        self._df = pd.DataFrame(columns=question_names)
        self.pane = pn.pane.DataFrame(self._df, sizing_mode="stretch_both", max_width=1000)

        self._last_response_t = None
    
    def update(self, t: int, data: np.ndarray, qa_state: QAState) -> None:
        latest_offset = qa_state.qa_offsets[-1]
        if self._last_response_t != latest_offset:
            self._last_response_t = latest_offset
            new_row = {qname: self._responses.get(qname, (None, None))[0] for qname in self._df.columns}
            self._df = pd.concat([self._df, pd.DataFrame([new_row])], ignore_index=True)
            self.pane.object = self._df

                
        
