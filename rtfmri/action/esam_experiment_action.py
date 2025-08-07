import time
from psychopy import event

from rtfmri.utils.log import get_logger
from rtfmri.action.experiment_action import ExperimentAction

log = get_logger()

class ESAMExperimentAction(ExperimentAction):
    def __init__(self, sync, gui=None):
        super().__init__(sync, gui)

    def _run(self) -> None:
        """Run the full GUI loop, including QA triggering and exit conditions, or sleep if no GUI."""
        if self.gui:
            self.gui.draw_resting_screen()
            if self.gui.test_latency:
                self.poll_trigger()
            if event.getKeys(['escape']):
                log.info('- User pressed escape key')
                self.sync.end.set()
            if self.sync.hit.is_set() and not self.gui.test_latency:
                responses = self.gui.run_full_QA()
                log.info(' - Responses: %s' % str(responses))
                self.sync.hit.clear()
                self.sync.qa_end.set()
        else:
            time.sleep(0.1)
    
    def _end(self) -> None:
        if self.test_latency:
            self.save_trigger()
        self.close_psychopy_infrastructure()