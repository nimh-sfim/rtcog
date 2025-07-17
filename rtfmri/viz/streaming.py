import threading
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas
import panel as pn
from holoviews.streams import Stream

from rtfmri.viz.score_plotter import ScorePlotter
from rtfmri.viz.streaming_config import StreamerConfig, SyncEvents, QAState

hv.extension('bokeh')
pn.extension()

def run_streamer(streamer_config, sync_events):
    streamer = Streamer(streamer_config, sync_events)
    streamer.run()

class Streamer:
    """Receives scores from Matcher and starts server to stream the data live"""
    def __init__(self, config: StreamerConfig, sync_events: SyncEvents):
        self.sync = sync_events
        self.sync.shm_ready.wait()
        
        Ntemplates = len(config.template_labels)
        self.mp_new_tr = self.sync.new_tr
        self.Nt = config.Nt
    
        self.match_scores = SharedMemory(name="match_scores")
        self.shared_arrs = {}
        self.shared_arrs["scores"] = np.ndarray((Ntemplates, self.Nt), dtype=np.float32, buffer=self.match_scores.buf)
        
        self.plotters = [ScorePlotter(config)] # Making this a list to extend in the future

        self.t = config.matching_opts.match_start
        self.vols_noqa = config.matching_opts.vols_noqa

        self.qa_onsets = []
        self.qa_offsets = []
        self.in_qa = False
        self.cooldown_end = None

    @property
    def qa_state(self):
        return QAState(
            self.qa_onsets,
            self.qa_offsets,
            self.in_qa,
            self.in_cooldown,
            self.cooldown_end
        )
    
    @property
    def in_cooldown(self):
        if self.qa_offsets:
            return self.t < self.qa_offsets[-1] + self.vols_noqa
        return False
    
    def update(self):
        # Wait for new data, then update plot
        try:
            while self.t < self.Nt:
                self.sync.new_tr.wait()
                self.sync.new_tr.clear()

                if self.sync.hit.is_set() and not self.in_qa:
                    self.qa_onsets.append(self.t)
                    self.in_qa = True
                elif self.sync.qa_end.is_set():
                    self.qa_offsets.append(self.t)
                    self.in_qa = False
                    self.cooldown_end = self.t + self.vols_noqa

                print(self.qa_state)
                for plotter in self.plotters:
                    plotter.update(self.t, self.shared_arrs[plotter.data_key][:,self.t], self.qa_state)
                
                self.t += 1
        finally:
            self._close_shared_memory()
    
    def run(self):
        threading.Thread(target=self.update, daemon=True).start()
        pn.serve(pn.Column(*(p.dmap for p in self.plotters)), start=True, show=True)
    
    def _close_shared_memory(self):
        self.match_scores.close()
        self.match_scores.unlink()
        
