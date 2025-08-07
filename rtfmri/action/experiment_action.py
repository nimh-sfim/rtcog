from rtfmri.utils.sync import SyncEvents

class ExperimentAction:
    def __init__(self, sync: SyncEvents, gui=None):
        self.sync = sync
        self.gui = gui
    
    def run(self) -> None:
        """Execute the action logic."""
        while not self.sync.end.is_set():
            self._run()
        self._end

    def _run(self):
        """Custom action logic to be implemented in subclasses."""
        raise NotImplementedError
    
    def _end(self):
        """Custom optional teardown logic to be implemented in subclasses."""
        pass