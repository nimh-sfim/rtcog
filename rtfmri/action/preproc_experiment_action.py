import time
from psychopy import event

from rtfmri.utils.log import get_logger
from rtfmri.action.experiment_action import ExperimentAction

log = get_logger()

class PreprocExperimentAction(ExperimentAction):
    def __init__(self, sync, gui=None):
        super().__init__(sync, gui)

    def _run(self):
        """
        Run the GUI loop until experiment ends.

        Parameters
        ----------
        sync : SyncEvents
            Multiprocessing event signals to monitor for ending the run.
        """
        if self.gui:
            self.gui.draw_resting_screen()
            if self.gui.test_latency:
                self.gui.poll_trigger()
            if event.getKeys(['escape']):
                log.info('- User pressed escape key')
                self.sync.end.set()
        else:
            time.sleep(0.1)
    
    def _end(self) -> None:
        if self.gui.test_latency:
            self.gui.save_trigger()
        self.gui.close_psychopy_infrastructure()