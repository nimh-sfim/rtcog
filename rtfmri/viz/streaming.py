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

def run_streamer(Nt, template_labels, match_start, mp_new_tr, mp_shm_ready, mp_evt_qa_end, mp_evt_hit):
    streamer = Streamer(Nt, template_labels, match_start, mp_new_tr, mp_shm_ready, mp_evt_qa_end, mp_evt_hit)
    streamer.run()

class Streamer:
    """Receives scores from Matcher and starts server to stream the data live"""
    def __init__(self, Nt, template_labels, match_start, mp_new_tr, mp_shm_ready, mp_evt_qa_end, mp_evt_hit):
        mp_shm_ready.wait()
        self.mp_evt_qa_end = mp_evt_qa_end
        self.mp_evt_hit = mp_evt_hit

        self.mp_new_tr = mp_new_tr
        self.Nt = Nt
        self.template_labels = template_labels
        self.Ntemplates = len(template_labels)

        self.t = match_start

        self.shm = SharedMemory(name="match_scores")
        self.shared_arr = np.ndarray((self.Ntemplates, Nt), dtype=np.float32, buffer=self.shm.buf)

        self.df = pd.DataFrame(np.nan, index=np.arange(self.Nt), columns=self.template_labels)

        self.next_stream = Stream.define('Next', t=int)(t=self.t)
        self.dmap = hv.DynamicMap(self.plot, streams=[self.next_stream])

    def update(self):
        try:
            while self.t < self.Nt:
                self.mp_new_tr.wait()
                self.mp_new_tr.clear()

                if self.mp_evt_hit.is_set():
                    self.qa_onsets.append(self.t)
                elif self.mp_evt_qa_end.is_set():
                    self.qa_offsets.append(self.t)

                self.update_df()
                self.t += 1
                self.next_stream.event(t=self.t)   # pass t explicitly here
        finally:
            self._close_shared_memory()

    def plot(self, t):
        df_plot = self.df.iloc[:t]
        curve = df_plot.hvplot.line(legend='top', width=1500)
        blocks = self.draw_blocks(t)
        
        x_max = max(t, 50)
        return (blocks * curve).opts(xlim=(0, x_max))

    def draw_blocks(self, t):
        height = 10
        blocks = []
        for onset, offset in zip(self.qa_onsets, self.qa_offsets):
            blocks.append((onset, 0, offset, height))
        if len(self.qa_onsets) > len(self.qa_offsets):
            current_onset = self.qa_onsets[-1]
            blocks.append((current_onset, 0, t, height))

        color = 'cyan' if self.mp_evt_qa_end.is_set() else 'blue'
        return hv.Rectangles(blocks).opts(alpha=0.2, color=color, line_color=None)

    def run(self):
        threading.Thread(target=self.update, daemon=True).start()
        pn.serve(self.dmap, start=True, show=True)

    def _close_shared_memory(self):
        # TODO: move to Matcher
        self.shm.close()
        self.shm.unlink()
        
