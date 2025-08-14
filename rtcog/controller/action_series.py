from abc import ABC
import multiprocessing as mp
from psychopy import event

from rtcog.utils.log import get_logger
from rtcog.gui.gui_utils import validate_likert_questions
from rtcog.gui.preproc_gui import PreprocGUI
from rtcog.gui.esam_gui import EsamGUI

log = get_logger()

class BaseActionSeries(ABC):
    def __init__(self, sync, *, opts=None, gui=None, **kwargs):
        self.sync = sync
        self.gui = gui
        self.opts = opts

        for k, v in kwargs.items():
            setattr(self, k, v)

    def on_start(self):
        """Run once before loop starts."""
        pass

    def on_loop(self):
        """Run every iteration of the controller loop."""
        pass

    def on_hit(self):
        """Triggered when sync.hit is set."""
        pass

    def on_end(self):
        """Run once after the loop ends."""
        pass


class PreprocActionSeries(BaseActionSeries):
    def __init__(self, sync, opts, gui=None, **kwargs):
        """Set up GUI"""
        if gui is None:
            gui = PreprocGUI(opts, **kwargs)
        super().__init__(sync, opts=opts, gui=gui)

    def on_start(self):
        """Draw resting screen"""
        self.gui.draw_resting_screen()
    
    def on_loop(self):
        """Poll for escape keys"""
        # TODO: fix bug that this won't shut down experiment
        if event.getKeys(['escape']):
            log.info('- User pressed escape key')
            self.sync.end.set()
        
    def on_end(self):
        """Shut down GUI"""
        self.gui.close_psychopy_infrastructure()


class ESAMActionSeries(PreprocActionSeries):
    def __init__(self, sync, opts):
        """Build shared responses based on Likert questions and set up GUI"""
        opts.likert_questions = validate_likert_questions(opts.q_path)
        manager = mp.Manager()
        shared_responses = manager.dict({q["name"]: (None, None) for q in opts.likert_questions})

        gui = EsamGUI(opts=opts, shared_responses=shared_responses)
        super().__init__(sync, opts=opts, gui=gui)

    def on_hit(self):
        """Collect voice recording and question responses"""
        responses = self.gui.run_full_QA()
        log.info(' - Responses: %s' % str(responses))
        self.sync.hit.clear()
        self.sync.action_end.set()

    
class LatencyTestActionSeries(PreprocActionSeries):
    def __init__(self, sync, opts, clock):
        super().__init__(sync, opts=opts, clock=clock)
    
    def on_loop(self):
        """Timestamp each trigger instance"""
        self.gui.poll_trigger()
    
    def on_end(self):
        """Save trigger timestamps and shut down GUI"""
        self.gui.save_trigger()
        self.gui.close_psychopy_infrastructure()
    