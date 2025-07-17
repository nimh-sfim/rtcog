import numpy as np
import pandas as pd
from multiprocessing.shared_memory import SharedMemory
import holoviews as hv
import hvplot.pandas
import panel as pn
from holoviews.streams import Stream

from rtfmri.viz.streaming_config import StreamerConfig, QAState

class ScorePlotter:
    """Receives scores from Matcher to stream the data live"""
    data_key = 'scores'
    def __init__(self, config: StreamerConfig):

        self.Nt = config.Nt
        self.template_labels = config.template_labels
        self.Ntemplates = len(config.template_labels)

        self.t = config.matching_opts.match_start
        self.hit_thr = config.hit_thr

        self.df = pd.DataFrame(np.nan, index=np.arange(self.Nt), columns=self.template_labels)
        self.dmap = hv.DynamicMap(self.plot, streams=[Stream.define('Next')()])
        
        self.polys_static = []
        
        self.qa_state = None
        
        self.last_cooldown_shown = None

    def update(self, t: int, data: np.ndarray, qa_state: QAState):
        self.t = t
        self.df.iloc[t] = data
        self.qa_state = qa_state
        self.dmap.event()

    def plot(self):
        line_plot = self.df.hvplot.line(legend='top', label='Match Scores', width=1500)
        overlays = [line_plot]

        if self.qa_state is None:
            return hv.Overlay(overlays)

        overlays.append(hv.HLine(self.hit_thr).opts(color='black', line_dash='dashed', line_width=1)) # Threshold line
        overlays.append(self._draw_hit_markers())

        if self.qa_state.in_qa:
            overlays.append(self._draw_dynamic_box())
        elif self.qa_state.qa_offsets and self.t == self.qa_state.qa_offsets[-1]:
            self.polys_static.append(self._draw_poly(self.qa_state.qa_onsets[-1], self.t).opts(alpha=0.2, color='blue', line_color=None))
        elif self.qa_state.in_cooldown:
            if self.qa_state.cooldown_end != self.last_cooldown_shown:
                self.polys_static.append(
                    self._draw_poly(self.qa_state.qa_offsets[-1], self.qa_state.cooldown_end)
                    .opts(alpha=0.2, color='cyan', line_color=None)
                )
                self.last_cooldown_shown = self.qa_state.cooldown_end
        
        overlays.append(hv.Overlay(self.polys_static) if self.polys_static else hv.Overlay([]))
        
        return hv.Overlay(overlays)

    def _draw_dynamic_box(self):
        return self._draw_poly(self.qa_state.qa_onsets[-1], self.t).opts(alpha=0.2, color='blue', line_color=None)
        
    def _draw_poly(self, start, end):
        return hv.Polygons([
            [(start, -5), (end, -5), (end, 10), (start, 10)]
        ])

    def _draw_hit_markers(self):
        points = []
        for hit_time in self.qa_state.qa_onsets:
            row = self.df.iloc[hit_time]
            if not row.isna().all():
                max_template = row.idxmax() # Template with the highest score
                max_score = row[max_template] # Highest score value
                points.append((hit_time, max_score, max_template))

        if points:
            df_points = pd.DataFrame(points, columns=["TR", "score", "template"])
            scatter = hv.Scatter(df_points, kdims=["TR"], vdims=["score", "template"]).opts(
                marker='circle',
                alpha=0.5,
                size=12,
                tools=['hover'],
                cmap='Category10'
            )
            return scatter
        else:
            return hv.Scatter([], kdims=["TR"], vdims=["score", "template"])


