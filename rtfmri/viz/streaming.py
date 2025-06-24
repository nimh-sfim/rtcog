import threading
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas
import panel as pn
from holoviews.streams import Stream

hv.extension('bokeh')
pn.extension()

def run_streamer(Nt, template_labels, match_start, mp_new_tr, mp_shm_ready):
    streamer = Streamer(Nt, template_labels, match_start, mp_new_tr, mp_shm_ready)
    streamer.run()

class Streamer:
    """Receives scores from Matcher and starts server to stream the data live"""
    def __init__(self, Nt, template_labels, match_start, mp_new_tr, mp_shm_ready):
        mp_shm_ready.wait()

        self.mp_new_tr = mp_new_tr
        self.Nt = Nt
        self.template_labels = template_labels
        self.Ntemplates = len(template_labels)

        self.t = match_start

        self.shm = SharedMemory(name="match_scores")
        self.shared_arr = np.ndarray((self.Ntemplates, Nt), dtype=np.float32, buffer=self.shm.buf)

        self.df = pd.DataFrame(np.nan, index=np.arange(self.Nt), columns=self.template_labels)
        self.dmap = hv.DynamicMap(self.plot, streams=[Stream.define('Next')()])

    def update_df(self):
        if self.t < self.Nt:
            self.df.iloc[self.t] = self.shared_arr[:, self.t]
            self.t += 1
    
    def update(self):
        # Wait for new data, then update plot
        try:
            while self.t < self.Nt:
                self.mp_new_tr.wait()
                self.mp_new_tr.clear()
                self.update_df()
                self.dmap.event()
        finally:
            self._close_shared_memory()
    
    def plot(self):
        return self.df.iloc[:self.t].hvplot.line(legend='top', label='Match Scores', width=1500)
    
    def run(self):
        threading.Thread(target=self.update, daemon=True).start()
        pn.serve(self.dmap, start=True, show=True)
    
    def _close_shared_memory(self):
        self.shm.close()
        self.shm.unlink()
        
