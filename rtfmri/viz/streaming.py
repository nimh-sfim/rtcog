import threading
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas
import panel as pn

from rtfmri.viz.score_plotter import ScorePlotter
from rtfmri.viz.map_plotter import MapPlotter
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
        self.tr_data = SharedMemory(name="tr_data")

        self.shared_arrs = {}
        self.shared_arrs["scores"] = np.ndarray((Ntemplates, self.Nt), dtype=np.float32, buffer=self.match_scores.buf)
        self.shared_arrs["tr_data"] = np.ndarray((config.Nv,), dtype=np.float32, buffer=self.tr_data.buf)
        
        self.plotters = [ScorePlotter(config), MapPlotter(config)] # Making this a list to extend in the future

        self.t = config.matching_opts.match_start
        self.vols_noqa = config.matching_opts.vols_noqa

        self.qa_onsets = []
        self.qa_offsets = []
        self.in_qa = False
        self.cooldown_end = None

    @property
    def qa_state(self) -> QAState:
        return QAState(
            self.qa_onsets,
            self.qa_offsets,
            self.in_qa,
            self.in_cooldown,
            self.cooldown_end
        )
    
    @property
    def in_cooldown(self) -> bool:
        if self.qa_offsets:
            return self.t < self.qa_offsets[-1] + self.vols_noqa
        return False
    
    def update(self) -> None:
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

                # TODO: make this less bad
                for plotter in self.plotters:
                    if plotter.data_key == "scores":
                        plot_data = self.shared_arrs["scores"][:, self.t]
                    elif plotter.data_key == "tr_data":
                        plot_data = self.shared_arrs["tr_data"]
                    else:
                        plot_data = None
                    plotter.update(self.t, plot_data, self.qa_state)
                
                self.t += 1
        finally:
            self._close_shared_memory()
    
    def run(self) -> None:
        threading.Thread(target=self.update, daemon=True).start()
        panels = []
        for p in self.plotters:
            if hasattr(p, 'dmap'):
                panels.append(p.dmap)
            elif hasattr(p, 'pane'):
                panels.append(p.pane)
        pn.serve(pn.Column(*panels), start=True, show=True)
    
    def _close_shared_memory(self) -> None:
        self.match_scores.close()
        self.match_scores.unlink()
        self.tr_data.close()
        
