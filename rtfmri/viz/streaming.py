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

def run_streamer(Nt, template_labels, match_start, vols_noqa, mp_new_tr, mp_shm_ready, mp_evt_qa_end, mp_evt_hit):
    streamer = Streamer(Nt, template_labels, match_start, vols_noqa, mp_new_tr, mp_shm_ready, mp_evt_qa_end, mp_evt_hit)
    streamer.run()

class Streamer:
    """Receives scores from Matcher and starts server to stream the data live"""
    def __init__(self, Nt, template_labels, match_start, vols_noqa, mp_new_tr, mp_shm_ready, mp_evt_qa_end, mp_evt_hit):
        mp_shm_ready.wait()

        self.mp_new_tr = mp_new_tr
        self.Nt = Nt
        self.template_labels = template_labels
        self.Ntemplates = len(template_labels)

        self.t = match_start
        self.vols_noqa = vols_noqa

        self.shm = SharedMemory(name="match_scores")
        self.shared_arr = np.ndarray((self.Ntemplates, Nt), dtype=np.float32, buffer=self.shm.buf)

        self.df = pd.DataFrame(np.nan, index=np.arange(self.Nt), columns=self.template_labels)
        self.dmap = hv.DynamicMap(self.plot, streams=[Stream.define('Next')()])
        
        self.mp_evt_qa_end = mp_evt_qa_end
        self.mp_evt_hit = mp_evt_hit
        
        self.qa_onsets = []
        self.qa_offsets = []
        self.in_qa = False
        self.in_cooldown = False

        self.polys_static = []

    def update_df(self):
        self.df.iloc[self.t] = self.shared_arr[:, self.t]
    
    def update(self):
        # Wait for new data, then update plot
        try:
            while self.t < self.Nt:
                self.mp_new_tr.wait()
                self.mp_new_tr.clear()


                if self.qa_offsets:
                    self.in_cooldown = self.t < self.qa_offsets[-1] + self.vols_noqa

                if self.mp_evt_hit.is_set() and not self.in_qa:
                    self.qa_onsets.append(self.t)
                    self.in_qa = True
                    # self.mp_evt_hit.clear()
                elif self.mp_evt_qa_end.is_set():
                    self.qa_offsets.append(self.t)
                    print(f"new static is {self.qa_onsets[-1]} to {self.t} long")
                    self.polys_static.append(self._draw_poly(self.qa_onsets[-1], self.t).opts(alpha=0.2, color='blue', line_color=None))
                    self.in_qa = False

                self.update_df()
                self.t += 1
                self.dmap.event()
        finally:
            self._close_shared_memory()
    

    def plot(self):
        line_plot = self.df.iloc[:self.t].hvplot.line(legend='top', label='Match Scores', width=1500)

        # Combine all static boxes
        static_polys = hv.Overlay(self.polys_static) if self.polys_static else hv.Overlay([])

        if self.in_qa:
            print(f"dynamic is {self.qa_onsets[-1]} to {self.t} long")
            qa_poly_dynamic = self._draw_poly(self.qa_onsets[-1], self.t).opts(alpha=0.2, color='blue', line_color=None)
            return line_plot * static_polys * qa_poly_dynamic
        if self.in_cooldown:
            wait_poly_dynamic = self._draw_poly(self.qa_offsets[-1], self.t).opts(alpha=0.2, color='cyan', line_color=None)
            return line_plot * static_polys * wait_poly_dynamic

        return line_plot * static_polys


    # def plot(self):
    #     line_plot = self.df.iloc[:self.t].hvplot.line(legend='top', label='Match Scores', width=1500)
        
    #     if self.in_qa:
    #         qa_poly_dynamic = self._draw_poly(self.qa_onsets[-1], self.t).opts(alpha=0.2, color='blue', line_color=None)
    #         return line_plot * self.polys_static * qa_poly_dynamic
        
    #     return line_plot * self.polys_static
      
    def run(self):
        threading.Thread(target=self.update, daemon=True).start()
        pn.serve(self.dmap, start=True, show=True)
    
    def _draw_poly(self, start, end):
        return hv.Polygons([
            [(start, -5), (end, -5), (end, 10), (start, 10)]
        ])

    def _close_shared_memory(self):
        self.shm.close()
        self.shm.unlink()
        
