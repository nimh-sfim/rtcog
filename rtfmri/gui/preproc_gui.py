import pickle
import os.path as osp

from psychopy import core, event
from psychopy import logging as psychopy_logging
from psychopy.visual import TextStim
from psychopy import prefs
prefs.hardware['keyboard'] = 'pygame'

from rtfmri.utils.core import get_logger
from rtfmri.gui.base_gui import BaseGUI

log = get_logger()

class PreprocGUI(BaseGUI):
    """
    Default GUI class for real-time fMRI experiments using PsychoPy.

    Displays a resting instruction screen and optionally collects TR trigger timings.

    Parameters
    ----------
    expInfo : dict
        Dictionary with display options, e.g., fullscreen and screen type.
    opts : Options
        Configuration options for the experiment run.
    clock : SharedClock, optional
        Clock object for precise timestamping of trigger events.
    """
    def __init__(self, expInfo, opts, clock=None, **kwargs):
        self.out_dir    = opts.out_dir
        self.out_prefix = opts.out_prefix

        if expInfo['fullScreen'] == 'Yes':
            self.fscreen = True
        else:
            self.fscreen = False

        if expInfo['screen'] == 'Laptop':
            self.screen = 0
        if expInfo['screen'] == 'External':
            self.screen = 1

        self.ewin = self._create_experiment_window()
        
        # Default Screen
        self.default_inst = [
            TextStim(win=self.ewin, text='Fixate on crosshair', pos=(0.0,0.42)),
            TextStim(win=self.ewin, text='Let your mind wander freely', pos=(0.0,0.3)),
            TextStim(win=self.ewin, text='Do not sleep', pos=(0.0,-0.3)),
            TextStim(win=self.ewin, text='X', pos=(0,0))
        ]
        
        self.test_latency = opts.test_latency
        self.clock = clock
        self.trigger_path = osp.join(opts.out_dir, f'{opts.out_prefix}_trigger_timing.pkl')
        self.triggers = []
  

    def draw_resting_screen(self):
        """
        Display the default resting instruction screen.
        """
        self._draw_stims(self.default_inst)
    
    def poll_trigger(self):
        """
        Listen for trigger keys ('t') and escape key.
        Records timestamps for triggers and allows early termination via `esc`.
        """
        keys = event.getKeys(['t', 'escape'])
        now = self.clock.now()
        for key in keys:
            if key == 't':
                self.triggers.append((now))
                print(f"Trig @ {now:.3f}            - Time point [t={len(self.triggers)}]", flush=True)
            elif key == 'escape':
                self.save_trigger()
                self.close_psychopy_infrastructure()

    def save_trigger(self):
        """
        Save collected trigger timestamps to disk.
        """
        with open(self.trigger_path, 'wb') as f:
            pickle.dump(self.triggers, f)
        print(f'Timing saved to {self.trigger_path}')

    def close_psychopy_infrastructure(self):
        """
        Cleanly close the PsychoPy display window and exit the experiment.
        """
        log.info('Closing psychopy window...')
        self.ewin.flip()
        self.ewin.close()
        psychopy_logging.flush()
        core.quit()
    