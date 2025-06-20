import pickle

from psychopy import core
from psychopy.iohub.client import launchHubServer
from psychopy.iohub.constants import EventConstants


class TriggerListener:
    def __init__(self, mp_evt_end, clock, trigger_path):
        self.mp_evt_end = mp_evt_end
        self.clock = clock
        self.out_path = trigger_path
        self.triggers = []  # initialize triggers here

    def capture_trigger(self):
        """Capture time of each scan acquisition (for latency testing purposes)"""
        self.triggers = []  # reset triggers list before capturing

        io = launchHubServer()
        keyboard = io.devices.keyboard
        print("TriggerListener: Ready to go")
        
        while not self.mp_evt_end.is_set():
            keys = keyboard.getKeys(keys=['t'], etype=EventConstants.KEYBOARD_RELEASE)
            if keys:
                for key in keys:
                    trigger_time = self.clock.now()
                    self.triggers.append(trigger_time)
                    print(f"Trig @ {trigger_time}", flush=True)
            core.wait(0.005)

        io.quit()
        self.save_triggers()

    def save_triggers(self):
        with open(self.out_path, 'wb') as f:
            pickle.dump(self.triggers, f)
        print(f'Timing saved to {self.out_path}')
