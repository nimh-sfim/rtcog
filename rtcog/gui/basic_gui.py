import pickle
import os
import os.path as osp

from psychopy import core, event
from psychopy import logging as psychopy_logging
from psychopy.visual import TextStim
from psychopy import prefs

if os.environ.get("READTHEDOCS") != "True":
    prefs.hardware['keyboard'] = 'pygame'

from rtcog.utils.core import get_logger
from rtcog.gui.gui_utils import get_experiment_info
from rtcog.gui.base_gui import BaseGUI

log = get_logger()

class BasicGUI(BaseGUI):
    """
    Default GUI class for real-time fMRI experiments using PsychoPy.

    Displays a resting instruction screen and optionally collects TR trigger timings.

    Parameters
    ----------
    opts : Options
        Configuration options for the experiment run.
    clock : SharedClock, optional
        Clock object for precise timestamping of trigger events.
    """
    def __init__(self, opts, *, clock=None, **kwargs):
        self.exp_info = get_experiment_info(opts)
        self.out_dir    = opts.out_dir
        self.out_prefix = opts.out_prefix

        if self.exp_info['fullScreen'] == 'Yes':
            self.fscreen = True
        else:
            self.fscreen = False

        if self.exp_info['screen'] == 'Laptop':
            self.screen = 0
        if self.exp_info['screen'] == 'External':
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
        if self.test_latency:
            self.ewin.winHandle.activate()
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
        keys = event.getKeys([self.exp_info["triggerKey"], 'escape'])
        now = self.clock.now()
        for key in keys:
            if key == 't':
                self.triggers.append((now))
                print(f"Trig @ {now:.3f}                 - Time point [t={len(self.triggers)}]", flush=True)
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
    