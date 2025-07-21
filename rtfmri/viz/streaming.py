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
from rtfmri.utils.log import get_logger

log = get_logger()

hv.extension('bokeh')
pn.extension()

def run_streamer(streamer_config, sync_events) -> None:
    streamer = Streamer(streamer_config, sync_events)
    streamer.run()

class Streamer:
    """Receives scores from Matcher and starts server to stream the data live"""
    def __init__(self, config: StreamerConfig, sync_events: SyncEvents):
        self._sync = sync_events
        self._sync.shm_ready.wait()
        
        Ntemplates = len(config.template_labels)
        self._Nt = config.Nt
    
        self._match_scores = SharedMemory(name="match_scores")
        self.tr_data = SharedMemory(name="tr_data")

        self._shared_arrs = {}
        self._shared_arrs["scores"] = np.ndarray((Ntemplates, self._Nt), dtype=np.float32, buffer=self._match_scores.buf)
        self._shared_arrs["tr_data"] = np.ndarray((config.Nv,), dtype=np.float32, buffer=self.tr_data.buf)
        
        self._plotters = [ScorePlotter(config), MapPlotter(config)] # Making this a list to extend in the future

        self.t = config.matching_opts.match_start
        self._vols_noqa = config.matching_opts.vols_noqa

        self._qa_onsets = []
        self._qa_offsets = []
        self._in_qa = False
        self._cooldown_end = None

    @property
    def qa_state(self) -> QAState:
        return QAState(
            self._qa_onsets,
            self._qa_offsets,
            self._in_qa,
            self.in_cooldown,
            self._cooldown_end
        )
    
    @property
    def in_cooldown(self) -> bool:
        if self._qa_offsets:
            return self.t < self._qa_offsets[-1] + self._vols_noqa
        return False
    
    def update(self) -> None:
        # Wait for new data, then update plot
        while not self._sync.end.is_set():
            self._sync.new_tr.wait()
            self._sync.new_tr.clear()

            if self._sync.hit.is_set() and not self._in_qa:
                self._qa_onsets.append(self.t)
                self._in_qa = True
            elif self._sync.qa_end.is_set():
                self._qa_offsets.append(self.t)
                self._in_qa = False
                self._cooldown_end = self.t + self._vols_noqa

            # TODO: make this less bad
            for plotter in self._plotters:
                if plotter.data_key == "scores":
                    plot_data = self._shared_arrs["scores"][:, self.t]
                elif plotter.data_key == "tr_data":
                    if self._qa_onsets and self.t == self._qa_onsets[-1]:
                        plot_data = self._shared_arrs["tr_data"]
                else:
                    plot_data = None
                if plot_data is not None:
                    plotter.update(self.t, plot_data, self.qa_state)
            
            self.t += 1

    def run(self) -> None:
        try:
            threading.Thread(target=self.update, daemon=True).start()

            panels = []
            for p in self._plotters:
                if hasattr(p, 'dmap'):
                    panels.append(p.dmap)
                elif hasattr(p, 'pane'):
                    panels.append(p.pane)

            self._server = pn.serve(
                pn.Column(*panels),
                start=True,
                show=True,
                threaded=True,
                port=5006
            )

            self._sync.end.wait()

        except Exception as e:
            log.error(e)

        finally:
            log.info("Shutting down Panel server...")
            if self._server:
                self._server.stop()
            self._close_shared_memory()

    
    def _close_shared_memory(self) -> None:
        self._match_scores.close()
        self._match_scores.unlink()
        self.tr_data.close()
        
