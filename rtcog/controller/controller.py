import time
from rtcog.utils.sync import SyncEvents
from rtcog.controller.action_series import BaseActionSeries

class Controller:
    def __init__(self, sync: SyncEvents, action_series: BaseActionSeries):
        self.sync = sync
        self.action_series = action_series
    
    def run(self):
        self.action_series.on_start()

        while not self.sync.end.is_set():
            self.action_series.on_loop()
            if self.sync.hit.is_set():
                self.action_series.on_hit()
            time.sleep(0.01)


        self.action_series.on_end()
