import time
import threading
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import ListProxy, DictProxy
import numpy as np
import holoviews as hv
import hvplot.pandas
import panel as pn

from rtfmri.utils.sync import SyncEvents, QAState
from rtfmri.viz.score_plotter import ScorePlotter
from rtfmri.viz.map_plotter import MapPlotter
from rtfmri.viz.response_plotter import ResponsePlotter
from rtfmri.viz.streaming_config import StreamingConfig
from rtfmri.utils.log import get_logger

log = get_logger()

hv.extension('bokeh')
pn.extension()

def run_streamer(streamer_config, sync_events, qa_onsets, qa_offsets, responses) -> None:
    """Instantiate streamer object and start thread."""
    streamer = ESAMStreamer(streamer_config, sync_events, qa_onsets, qa_offsets, responses)
    streamer.run()

class ESAMStreamer:
    """
    Streamer for realtime fMRI data visualization.

    This class receives shared memory updates and streams the data live using
    Panel-based visualizations. It streams three plots:

        - ScorePlotter (match scores)
        - MapPlotter (activation maps)
        - ResponsePlotter (behavioral responses)

    Parameters
    ----------
    config : StreamingConfig
        Configuration object containing information about the fMRI session.
    sync_events : SyncEvents
        Object that handles interprocess synchronization flags and signals.
    qa_onsets : ListProxy
        Shared list storing the TR indices of QA onsets (trial starts).
    qa_offsets : ListProxy
        Shared list storing the TR indices of QA offsets (trial ends).
    responses : DictProxy
        Shared dictionary containing responses collected during the experiment.
    """
    def __init__(self, config: StreamingConfig, sync_events: SyncEvents, qa_onsets: ListProxy, qa_offsets: ListProxy, responses: DictProxy):
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
        
        self._plotters = [ScorePlotter(config), MapPlotter(config), ResponsePlotter(config, responses)]

        self._vols_noqa = config.matching_opts.vols_noqa

        self._qa_onsets = qa_onsets
        self._qa_offsets = qa_offsets
        self._in_qa = False
        self._cooldown_end = None
        self._hit = False

    @property
    def qa_state(self) -> QAState:
        """
        Construct and return the current QAState.

        Returns
        -------
        QAState
            An object encapsulating current trial timing state.
        """
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
        """
        Check whether the streamer is currently in a post-trial cooldown period.

        Returns
        -------
        bool
            True if in cooldown, False otherwise.
        """
        if self._qa_offsets:
            return self._last_t < self._qa_offsets[-1] + self._vols_noqa
        return False
    
    def update(self) -> None:
        """
        Method run in a background thread to stream new data.

        Waits for new TRs and passes the appropriate data to each plotter.
        """
        while not self._sync.end.is_set():
            t = self._last_t + 1

            # Wait for new data for next TR, or sleep briefly
            if t >= self._Nt or t > self._sync.tr_index.value:
                time.sleep(0.01)
                continue

            # Now data for t is available, so increment and process
            self._last_t = t
            self._update_qa_state(t)

            for plotter in self._plotters:
                if not plotter.should_update(t, self.qa_state):
                    continue

                data = None
                key = plotter.data_key
                if key in self._shared_arrs:
                    data = self._shared_arrs[key][:, t]
                
                plotter.update(t, data, self.qa_state)


    def run(self) -> None:
        """
        Start the background update thread and launch the Panel visualization server.
        """
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
            self._sync.server_ready.set()

            self._sync.end.wait()

        finally:
            self._shutdown()
            
    def _update_qa_state(self, t: int) -> None:
        """
        Update internal QA state based on the current TR.

        Parameters
        ----------
        t : int
            Current TR index.
        """
        self._hit = False
        if t in self._qa_onsets and not self._in_qa:
            self._hit = True
            self._in_qa = True
        elif self._qa_offsets and t in self._qa_offsets:
            self._in_qa = False
            self._cooldown_end = t + self._vols_noqa

    def _shutdown(self) -> None:
        """
        Stop the server, close shared memory, and shut down all plotters.
        """
        log.info("Shutting down Panel server...")
        if self._server:
            self._server.stop()
        self._close_shared_memory()
        for p in self._plotters:
            p.close()

    def _close_shared_memory(self) -> None:
        """
        Close and unlink shared memory segments.
        """
        self._match_scores.close()
        self._match_scores.unlink()
        self._tr_data.close()
        
