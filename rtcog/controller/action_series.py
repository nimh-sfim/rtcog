import os.path as osp
from abc import ABC
import multiprocessing as mp
from psychopy import event
import pandas as pd
import matplotlib.pyplot as plt

from rtcog.utils.log import get_logger
from rtcog.gui.gui_utils import validate_likert_questions
from rtcog.gui.base_gui import BaseGUI
from rtcog.gui.basic_gui import BasicGUI
from rtcog.gui.esam_gui import EsamGUI
from rtcog.utils.core import SharedClock
from rtcog.utils.sync import SyncEvents
from rtcog.utils.options import Options

log = get_logger()

class BaseActionSeries(ABC):
    """
    Base class defining methods that are called by the Controller during an experiment.

    The Controller manages the experiment flow and calls these methods at different
    stages, such as at the start of the experiment, during each loop iteration,
    when a hit event occurs, and at the end. Subclasses can override these methods
    to implement task-specific behavior.
    """
    def __init__(self, sync: SyncEvents, *, opts: Options=None, gui: BaseGUI=None, **kwargs):
        """
        Initialize the ActionSeries.

        Parameters
        ----------
        sync : SyncEvents
            Shared synchronization events used to coordinate experiment state.
        opts : Options, optional
            Options object containing experiment settings.
        gui : BaseGUI, optional
            GUI instance used by the action series.
        **kwargs
            Additional attributes to attach to the instance.
        """
        self.sync = sync
        self.gui = gui
        self.opts = opts

        # Attach additional attributes to the instance
        for k, v in kwargs.items():
            setattr(self, k, v)

    def on_start(self):
        """Called once at the beginning of the experiment for initialization."""
        pass

    def on_loop(self):
        """Called every iteration of the controller loop."""
        pass

    def on_hit(self):
        """Called whenever a hit event occurs during the experiment."""
        pass

    def on_end(self):
        """Called once after the loop ends."""
        pass


class BasicActionSeries(BaseActionSeries):
    """
    Action series providing basic GUI handling.

    This class displays a resting screen and monitors for
    user-triggered experiment termination.
    """
    def __init__(self, sync: SyncEvents, opts: Options, gui: BasicGUI=None, **kwargs):
        """
        Initialize the preprocessing action series.

        Parameters
        ----------
        sync : SyncEvents
            Shared synchronization events.
        opts : Options
            Options object containing experiment settings.
        gui : BasicGUI, optional
            GUI instance. If not provided, a BasicGUI is created.
        **kwargs
            Additional arguments passed to the GUI constructor.
        """
        if gui is None:
            gui = BasicGUI(opts, **kwargs)
        super().__init__(sync, opts=opts, gui=gui)

    def on_start(self):
        """
        Draw the resting screen at experiment start.
        """
        self.gui.draw_resting_screen()
    
    def on_loop(self):
        """
        Poll for user escape key presses.

        If the escape key is detected, the global end event is set to signal
        experiment termination.
        """
        # TODO: fix bug that this won't shut down experiment
        if event.getKeys(['escape']):
            log.info('- User pressed escape key')
            self.sync.end.set()
        
    def on_end(self):
        """
        Shut down GUI at experiment termination.
        """
        self.gui.close_psychopy_window()


class ESAMActionSeries(BasicActionSeries):
    """
    Action series implementing ESAM task behavior.

    This series extends preprocessing behavior by collecting voice recordings
    and Likert-scale responses during action periods and saving them
    at experiment end.
    """
    def __init__(self, sync: SyncEvents, opts: Options):
        """
        Initialize the ESAM action series.

        This method validates Likert questions, creates shared response storage,
        and initializes the ESAM GUI.

        Parameters
        ----------
        sync : SyncEvents
            Shared synchronization events.
        opts : Options
            Configuration options, including Likert question definitions.
        """
        opts.likert_questions = validate_likert_questions(opts.q_path)

        # Shared dictionary to store responses across processes
        manager = mp.Manager()
        shared_responses = manager.dict({q["name"]: (None, None) for q in opts.likert_questions})

        gui = EsamGUI(opts=opts, shared_responses=shared_responses)
        super().__init__(sync, opts=opts, gui=gui)

    def on_hit(self):
        """
        Handle action onset by collecting recordings and questionnaire responses.

        This method:
        - Runs the full action interaction
        - Logs rounded response values
        - Clears the hit event
        - Signals action completion
        - Returns the GUI to the resting screen
        """
        responses = self.gui.run_full_action()

        rounded_responses = {
            k: (v[0], round(v[1], 2)) for k, v in responses.items()
        }
        log.debug(f' - Responses: {rounded_responses}')

        # Reset hit event and mark action completion
        self.sync.hit.clear()
        self.sync.action_end.set()

        self.gui.draw_resting_screen()
    
    def on_end(self):
        """
        Save Likert responses and shut down GUI resources.
        """
        self.gui.save_likert_files()
        self.gui.close_psychopy_window()

    
class LatencyTestActionSeries(BasicActionSeries):
    """
    Action series used for trigger latency testing.

    This series timestamps trigger events during the controller loop and
    saves timing information at experiment end.
    """
    def __init__(self, sync: SyncEvents, opts: Options, clock: SharedClock):
        """
        Initialize the latency test action series.

        Parameters
        ----------
        sync : SyncEvents
            Shared synchronization events.
        opts : Options
            Options object containing experiment settings.
        clock : object
            Clock to timestamp triggers.
        """
        super().__init__(sync, opts=opts, clock=clock)

    
    def on_loop(self):
        """
        Poll for trigger events and record their timestamps.
        """
        self.gui.poll_trigger()
    
    def on_end(self):
        """
        Save trigger timestamps, print metrics, and shut down GUI
        """
        self.sync.hit.set()  # Ensure process waits for action end
        self.gui.save_trigger()
        self.gui.close_psychopy_window()
        df = self._calculate_latency_metrics()
        if df is not None:
            self._graph_latency(df)
        self.sync.hit.clear()
    
    def _calculate_latency_metrics(self) -> pd.DataFrame | None:
        """
        Print latency metrics to the log and create df for graphing.
        
        Discarded TRs are removed before calculation.
        
        Returns
        -------
        pd.DataFrame or None
            DataFrame containing latency metrics, or None if missing triggers.
        """
        trig_path = osp.join(f"{self.opts.out_prefix}_trigger_timing.pkl")
        rec_path = osp.join(f"{self.opts.out_prefix}_receiver_timing.pkl")
        
        trig = pd.read_pickle(trig_path)
        rec = pd.read_pickle(rec_path)
        
        df = pd.DataFrame.from_dict(rec)

        if len(df) != self.opts.nvols:
            log.warning(f"Number of received TRs ({len(df)}) does not match expected ({self.opts.nvols})")

        if len(trig) == len(df):
            df.insert(0, 'trig', trig)

        # Drop discarded TRs
        df = df.iloc[self.opts.discard:].reset_index(drop=True)
        if 'trig' in df.columns:
            trig = trig[self.opts.discard:]

        df.index = range(self.opts.discard, self.opts.discard + len(df))

        mean_compute_time =  (df['proc'] - df['recv']).mean()
        log.info(f"Mean time between recv and processing: {mean_compute_time:.3f} seconds")

        if 'trig' in df.columns:
            mean_latency_recv = (df['recv'] - df['trig']).mean()
            log.info(f"Mean time between trigger and recv: {mean_latency_recv:.3f} seconds")

            mean_latency_proc = (df['proc'] - df['trig']).mean()
            log.info(f"Mean time between trigger and processing: {mean_latency_proc:.3f} seconds")
            return df

        else: 
            log.warning(f"Number of triggers ({len(trig)}) does not match number of received TRs ({len(df)})")
            return None
        

    def _graph_latency(self, df):
        """
        Graph latency metrics and save to disk.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing latency metrics.
        """
        df_rel = pd.DataFrame({
            't_trig': 0,  # always zero since metrics are relative to trigger
            't_recv': df['recv'] - df['trig'],
            't_proc': df['proc'] - df['trig']
        })
        
        plt.figure(figsize=(12, 6))

        plt.scatter(df_rel.index, df_rel['t_trig'], label='Trigger', color='blue', marker='o')
        plt.scatter(df_rel.index, df_rel['t_recv'], label='Received', color='orange', marker='o')
        plt.scatter(df_rel.index, df_rel['t_proc'], label='Processed', color='green', marker='o')
        
        plt.axvline(x=self.opts.discard, color='red', linestyle='--', label='Discard Boundary')
        
        plt.xlabel('TR')
        plt.ylabel('Time Since Trigger (seconds)')
        plt.title(f'{self.opts.out_prefix}: Latency Relative to Trigger')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        out = f"{self.opts.out_prefix}_latency.png"
        plt.savefig(out, dpi=300)
        plt.close()
        
        log.info(f"Latency graph saved to {out}")