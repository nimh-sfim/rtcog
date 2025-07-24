import os.path as osp
from itertools import cycle
import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas
import panel as pn
from holoviews.streams import Stream
from bokeh.palettes import Category10

from rtfmri.utils.sync import QAState
from rtfmri.viz.streaming_config import StreamingConfig

class ScorePlotter:
    """Receives scores from Matcher to stream the data live"""
    data_key = 'scores'
    def __init__(self, config: StreamingConfig):
        self._Nt = config.Nt
        self._template_labels = config.template_labels
        self._Ntemplates = len(self._template_labels)

        self._hit_thr = config.hit_thr

        self._df = pd.DataFrame(np.nan, index=np.arange(self._Nt), columns=self._template_labels)
        self.dmap = hv.DynamicMap(self._plot, streams=[Stream.define('Next', t=int)()])
        
        self._polys_static = []
        self._no_match_poly = self._draw_poly(0, config.matching_opts.match_start).opts(color='gray', line_color=None, alpha=0.2)
        self._qa_state = None
        
        self._last_cooldown_shown = None
        
        self._colors = self._get_template_colors()
        
        self._out_prefix = config.out_prefix
        self._out_dir = config.out_dir
        
    def update(self, t: int, data: np.ndarray, qa_state: QAState) -> None:
        self._df.iloc[t] = data
        self._qa_state = qa_state
        if not np.isnan(self._df.iloc[t]).all():
            self.dmap.event(t=t)

    def _plot(self, t: int) -> hv.Overlay:
        line_plot = self._df.hvplot.line(legend='top', label='Match Scores', width=1500, cmap=self._colors)
        overlays = [line_plot, self._no_match_poly]

        if self._qa_state is None:
            return hv.Overlay(overlays)

        overlays.append(hv.HLine(self._hit_thr).opts(color='black', line_dash='dashed', line_width=1)) # Threshold line
        overlays.append(self._draw_hit_markers())

        if self._qa_state.in_qa:
            overlays.append(self._draw_dynamic_box(t))
        elif self._qa_state.qa_offsets and t == self._qa_state.qa_offsets[-1]:
            self._polys_static.append(self._draw_poly(self._qa_state.qa_onsets[-1], t).opts(alpha=0.2, color='blue', line_color=None))
        elif self._qa_state.in_cooldown:
            if self._qa_state.cooldown_end != self._last_cooldown_shown:
                self._polys_static.append(
                    self._draw_poly(self._qa_state.qa_offsets[-1], self._qa_state.cooldown_end)
                    .opts(alpha=0.2, color='cyan', line_color=None)
                )
                self._last_cooldown_shown = self._qa_state.cooldown_end
        
        overlays.append(hv.Overlay(self._polys_static) if self._polys_static else hv.Overlay([]))
        
        return hv.Overlay(overlays)

    def _draw_dynamic_box(self, t: int) -> hv.Polygons:
        return self._draw_poly(self._qa_state.qa_onsets[-1], t).opts(alpha=0.2, color='blue', )
        
    def _draw_poly(self, start: int, end: int) -> hv.Polygons:
        end = min(end, self._Nt)
        return hv.Polygons([
            [(start, -5), (end, -5), (end, 5), (start, 5)]
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
            df_points['color'] = df_points['template'].map(self._colors).fillna('gray')

            return hv.Scatter(df_points, kdims=["TR"], vdims=["score", "template", "color"]).opts(
                marker='circle',
                alpha=0.5,
                size=12,
                tools=['hover'],
                color='color'
            )
        else:
            return hv.Scatter([], kdims=["TR"], vdims=["score", "template"])
    
    def _get_template_colors(self):
        palette = Category10[10]

        # Cycle through palette if more labels than colors
        assigned_colors = [c for _, c in zip(self._template_labels, cycle(palette))]
        return dict(zip(self._template_labels, assigned_colors))
    
    def close(self):
        """Save the final state of the streaming plot to an HTML file"""
        out_html = osp.join(self._out_dir, self._out_prefix + '.dyn_report.html')
        renderer = hv.renderer('bokeh')

        # Get last time index with valid data
        last_valid_idx = self._df.dropna(how='all').index.max()
        final_plot = self._plot(last_valid_idx)

        renderer.save(final_plot, out_html)
        print(f'++ Dynamic Report written to disk: [{out_html}]')