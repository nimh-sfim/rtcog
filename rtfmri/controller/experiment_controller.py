from rtfmri.utils.sync import SyncEvents

class ExperimentController:
    def __init__(self, sync: SyncEvents, gui=None):
        self.sync = sync
        self.gui = gui
    
    def run(self) -> None:
        """Execute the controller logic."""
        while not self.sync.end.is_set():
            self._run()
        self._end

    def _run(self):
        """Custom controller logic to be implemented in subclasses."""
        raise NotImplementedError
    
    def _end(self):
        """Custom optional teardown logic to be implemented in subclasses."""
        pass