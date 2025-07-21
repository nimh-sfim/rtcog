import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas
import panel as pn
from holoviews.streams import Stream

from rtfmri.viz.streaming_config import StreamerConfig, QAState

class ScorePlotter:
    """Receives scores from Matcher to stream the data live"""
    data_key = 'scores'
    def __init__(self, config: StreamerConfig):
        self._Nt = config.Nt
        self._template_labels = config.template_labels
        self._Ntemplates = len(config.template_labels)

        self._t = config.matching_opts.match_start
        self._hit_thr = config.hit_thr

        self._df = pd.DataFrame(np.nan, index=np.arange(self._Nt), columns=self._template_labels)
        self.dmap = hv.DynamicMap(self._plot, streams=[Stream.define('Next')()])
        
        self._polys_static = []
        
        self._qa_state = None
        
        self._last_cooldown_shown = None

    def update(self, t: int, data: np.ndarray, qa_state: QAState) -> None:
        self._t = t
        self._df.iloc[t] = data
        self._qa_state = qa_state
        self.dmap.event()

    def _plot(self) -> hv.Overlay:
        line_plot = self._df.hvplot.line(legend='top', label='Match Scores', width=1500)
        overlays = [line_plot]

        if self._qa_state is None:
            return hv.Overlay(overlays)

        overlays.append(hv.HLine(self._hit_thr).opts(color='black', line_dash='dashed', line_width=1)) # Threshold line
        overlays.append(self._draw_hit_markers())

        if self._qa_state.in_qa:
            overlays.append(self._draw_dynamic_box())
        elif self._qa_state.qa_offsets and self._t == self._qa_state.qa_offsets[-1]:
            self._polys_static.append(self._draw_poly(self._qa_state.qa_onsets[-1], self._t).opts(alpha=0.2, color='blue', line_color=None))
        elif self._qa_state.in_cooldown:
            if self._qa_state.cooldown_end != self._last_cooldown_shown:
                self._polys_static.append(
                    self._draw_poly(self._qa_state.qa_offsets[-1], self._qa_state.cooldown_end)
                    .opts(alpha=0.2, color='cyan', line_color=None)
                )
                self._last_cooldown_shown = self._qa_state.cooldown_end
        
        overlays.append(hv.Overlay(self._polys_static) if self._polys_static else hv.Overlay([]))
        
        return hv.Overlay(overlays)

    def _draw_dynamic_box(self) -> hv.Polygons:
        return self._draw_poly(self._qa_state.qa_onsets[-1], self._t).opts(alpha=0.2, color='blue', line_color=None)
        
    def _draw_poly(self, start, end) -> hv.Polygons:
        return hv.Polygons([
            [(start, -5), (end, -5), (end, 10), (start, 10)]
        ])

    def _draw_hit_markers(self) -> hv.Scatter:
        points = []
        for hit_time in self._qa_state.qa_onsets:
            row = self._df.iloc[hit_time]
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


