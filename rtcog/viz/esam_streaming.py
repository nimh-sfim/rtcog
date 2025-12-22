import time
import threading
import os.path as osp
from rtcog.utils.shared_memory_manager import SharedMemoryManager
from multiprocessing.managers import ListProxy, DictProxy
import numpy as np
import holoviews as hv
import panel as pn

from rtcog.utils.sync import SyncEvents, ActionState
from rtcog.viz.score_plotter import ScorePlotter
from rtcog.viz.map_plotter import MapPlotter
from rtcog.viz.response_plotter import ResponsePlotter
from rtcog.viz.streaming_config import StreamingConfig
from rtcog.paths import RESOURCES_DIR
from rtcog.utils.log import get_logger

log = get_logger()

hv.extension('bokeh')
pn.extension()

def run_streamer(streamer_config, sync_events, action_onsets, action_offsets, responses) -> None:
    """Instantiate streamer object and start thread."""
    streamer = ESAMStreamer(streamer_config, sync_events, action_onsets, action_offsets, responses)
    streamer.run()

class ESAMStreamer:
    """
    Streamer for realtime fMRI data visualization.

    This class receives shared memory updates and streams the data live using
    a Panel server. It streams three plots:

        - ScorePlotter (match scores)
        - MapPlotter (activation maps)
        - ResponsePlotter (behavioral responses)

    Parameters
    ----------
    config : StreamingConfig
        Configuration object containing information about the fMRI session.
    sync_events : SyncEvents
        Object that handles interprocess synchronization flags and signals.
    action_onsets : ListProxy
        Shared list storing the TR indices of action onsets (trial starts).
    action_offsets : ListProxy
        Shared list storing the TR indices of action offsets (trial ends).
    responses : DictProxy
        Shared dictionary containing responses collected during the experiment.
    """
    def __init__(self, config: StreamingConfig, sync_events: SyncEvents, action_onsets: ListProxy, action_offsets: ListProxy, responses: DictProxy):
        self._sync = sync_events
        self._sync.shm_ready.wait()
        
        self._last_t = config.matching_opts.match_start - 1
        
        Ntemplates = len(config.template_labels)
        self._Nt = config.Nt
    
        self._match_scores = SharedMemoryManager("match_scores")
        self._tr_data = SharedMemoryManager("tr_data")

        self._match_scores_shm = self._match_scores.open()
        self._tr_data_shm = self._tr_data.open()

        self._shared_arrs = {}
        self._shared_arrs["scores"] = np.ndarray((Ntemplates, self._Nt), dtype=np.float32, buffer=self._match_scores_shm.buf)
        self._shared_arrs["tr_data"] = np.ndarray((config.Nv, config.Nt), dtype=np.float32, buffer=self._tr_data_shm.buf)
        
        self._plotters = [ScorePlotter(config), MapPlotter(config)]
        if responses is not None:
            self._plotters.append(ResponsePlotter(config, responses))

        self._vols_noaction = config.matching_opts.vols_noaction

        self._action_onsets = action_onsets
        self._action_offsets = action_offsets
        self._in_action = False
        self._cooldown_end = None
        self._hit = False

    @property
    def action_state(self) -> ActionState:
        """
        Construct and return the current ActionState.

        Returns
        -------
        ActionState
            An object encapsulating current trial timing state.
        """
        return ActionState(
            list(self._action_onsets),
            list(self._action_offsets),
            self._in_action,
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
        if self._action_offsets:
            return self._last_t < self._action_offsets[-1] + self._vols_noaction
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
            self._sync.new_tr.wait(timeout=0.05)


            # Now data for t is available, so increment and process
            self._last_t = t
            self._update_action_state(t)

            for plotter in self._plotters:
                if not plotter.should_update(t, self.action_state):
                    continue

                data = None
                key = plotter.data_key
                if key in self._shared_arrs:
                    data = self._shared_arrs[key][:, t]
                
                plotter.update(t, data, self.action_state)


    def run(self) -> None:
        """
        Start the background update thread and launch the Panel visualization server.
        """
        try:
            threading.Thread(target=self.update, daemon=True).start()

            panels = [
                getattr(p, 'dmap', getattr(p, 'pane', None))
                for p in self._plotters
                if hasattr(p, 'dmap') or hasattr(p, 'pane')
            ]

            template = pn.template.FastListTemplate(
                title="Live Data",
                logo=osp.join(RESOURCES_DIR, "brain_white.png"),
                theme_toggle=False,
            )
            template.main.extend(panels)

            self._server = pn.serve(
                template,
                start=True,
                show=True,
                threaded=True,
                port=5006
            )
            self._sync.server_ready.set()

            self._sync.end.wait()

        finally:
            self._shutdown()
            
    def _update_action_state(self, t: int) -> None:
        """
        Update internal action state based on the current TR.

        Parameters
        ----------
        t : int
            Current TR index.
        """
        self._hit = False
        if t in self._action_onsets and not self._in_action:
            self._hit = True
            self._in_action = True
        elif self._action_offsets and t in self._action_offsets:
            self._in_action = False
            self._cooldown_end = t + self._vols_noaction

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
        if hasattr(self, '_match_scores'):
            self._match_scores.cleanup()
        if hasattr(self, '_tr_data'):
            self._tr_data.cleanup()
