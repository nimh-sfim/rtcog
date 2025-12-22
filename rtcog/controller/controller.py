import time
from rtcog.utils.sync import SyncEvents
from rtcog.controller.action_series import BaseActionSeries

class Controller:
    """
    Orchestrates the execution of an action series based on synchronization events.

    The Controller class is responsible for managing the lifecycle of an action series.
    It listens for synchronization events and triggers the appropriate hooks in the
    action series in response.
    """
    def __init__(self, sync: SyncEvents, action_series: BaseActionSeries):
        """
        Initialize the Controller.

        Parameters
        ----------
        sync : SyncEvents
            Multiprocessing event signals for synchronization.
        action_series : BaseActionSeries
            Action series instance defining hooks for different runtime states
            (start, loop, hit, end).
        """
        self.sync = sync
        self.action_series = action_series
    
    def run(self):
        """
        Run the controller loop.

        This method:

        - Calls `on_start()` once at initialization

        - Repeatedly calls `on_loop()` until the end event is set

        - Triggers `on_hit()` whenever a hit event is detected

        - Calls `on_end()` once before exiting
        """
        self.action_series.on_start()

        while not self.sync.end.is_set():
            self.action_series.on_loop()
            if self.sync.hit.is_set():
                self.action_series.on_hit()
            time.sleep(0.01)

        self.action_series.on_end()
