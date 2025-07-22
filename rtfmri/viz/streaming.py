import time
import threading
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import ListProxy
import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas
import panel as pn

from rtfmri.utils.sync import SyncEvents, QAState
from rtfmri.viz.score_plotter import ScorePlotter
from rtfmri.viz.map_plotter import MapPlotter
from rtfmri.viz.streaming_config import StreamingConfig
from rtfmri.utils.log import get_logger

log = get_logger()

hv.extension('bokeh')
pn.extension()

def run_streamer(streamer_config, sync_events, qa_onsets, qa_offsets) -> None:
    streamer = Streamer(streamer_config, sync_events, qa_onsets, qa_offsets)
    streamer.run()

class Streamer:
    """Receives scores from Matcher and starts server to stream the data live"""
    def __init__(self, config: StreamingConfig, sync_events: SyncEvents, qa_onsets: ListProxy, qa_offsets: ListProxy):
        self._sync = sync_events
        self._sync.shm_ready.wait()
        
        self._last_t = config.matching_opts.match_start - 1
        
        Ntemplates = len(config.template_labels)
        self._Nt = config.Nt
    
        self._match_scores = SharedMemory(name="match_scores")
        tr_data = SharedMemory(name="tr_data")
        self._tr_data = tr_data

        self._shared_arrs = {}
        self._shared_arrs["scores"] = np.ndarray((Ntemplates, self._Nt), dtype=np.float32, buffer=self._match_scores.buf)
        self._shared_arrs["tr_data"] = np.ndarray((config.Nv, config.Nt), dtype=np.float32, buffer=tr_data.buf)
        
        self._plotters = [ScorePlotter(config), MapPlotter(config)] # Making this a list to extend in the future

        self._vols_noqa = config.matching_opts.vols_noqa

        self._qa_onsets = qa_onsets
        self._qa_offsets = qa_offsets
        self._in_qa = False
        self._cooldown_end = None
        self._hit = False

    @property
    def qa_state(self) -> QAState:
        return QAState(
            list(self._qa_onsets),
            list(self._qa_offsets),
            self._in_qa,
            self.in_cooldown,
            self._cooldown_end,
            self._hit
        )
    
    @property
    def in_cooldown(self) -> bool:
        if self._qa_offsets:
            return self._last_t < self._qa_offsets[-1] + self._vols_noqa
        return False
    
    def update(self) -> None:
        # Wait for new data, then update plot
        while not self._sync.end.is_set():
            t = self._last_t + 1  # check next index

            # Wait for new data for next TR, or sleep briefly
            if t >= self._Nt or t > self._sync.tr_index.value:
                time.sleep(0.01)
                continue

            # Now data for t is available, so increment and process
            self._last_t = t

            self._update_qa_state(t)

            # TODO: make this less bad
            for plotter in self._plotters:
                plot_data = None
                if plotter.data_key == "scores":
                    plot_data = self._shared_arrs["scores"][:, t]
                elif plotter.data_key == "tr_data":
                    if self._qa_onsets and t in self._qa_onsets:
                        print(f"Sharing new data @ {t}")
                        plot_data = self._shared_arrs["tr_data"][:, t]
                        print(f'[streamer] sum of the data: {plot_data.sum()}')
                        np.save(f"reader_tr_{t}.npy", plot_data.copy())
                if plot_data is not None:
                    plotter.update(t, plot_data, self.qa_state)

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

        finally:
            self._shutdown()
            
    def _update_qa_state(self, t: int) -> None:
        self._hit = False
        if t in self._qa_onsets and not self._in_qa:
            self._hit = True
            self._in_qa = True
        elif self._qa_offsets and t in self._qa_offsets:
            self._in_qa = False
            self._cooldown_end = t + self._vols_noqa

    def _shutdown(self) -> None:
        log.info("Shutting down Panel server...")
        if self._server:
            self._server.stop()
        self._close_shared_memory()
        for p in self._plotters:
            if hasattr(p, 'close'):
                p.close()

    def _close_shared_memory(self) -> None:
        self._match_scores.close()
        self._match_scores.unlink()
        self._tr_data.close()
        
