import os.path as osp
from itertools import cycle
import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas
import panel as pn
from holoviews.streams import Stream
from bokeh.palettes import Category10

from rtcog.utils.sync import ActionState
from rtcog.viz.streaming_config import StreamingConfig
from rtcog.viz.plotter import Plotter

class ScorePlotter(Plotter):
    """
    Live and post-hoc visualization of template matching scores.

    This plotter receives template matching scores over time and visualizes them
    as streaming line plots. It overlays action-related annotations including hit
    thresholds, action windows, cooldown periods, and hit markers indicating the
    strongest-matching template at action onset times.

    The plot can be rendered dynamically during acquisition or saved as a static
    HTML report at the end of the experiment.
    """
    data_key = 'scores'

    def __init__(self, config: StreamingConfig, streaming=True):
        """
        Initialize the ScorePlotter.

        Parameters
        ----------
        config : StreamingConfig
            Configuration object containing plotting, streaming, and action parameters.
        streaming : bool, optional
            If True, enables live updating via a HoloViews DynamicMap.
            If False, the plotter is used only for static rendering.
        """
        super().__init__(config)
        self._hit_thr = config.hit_thr
        self._match_start = config.matching_opts.match_start
        self._vols_noaction = config.matching_opts.vols_noaction
        self._y_range = self._threshold_y_range()

        # DataFrame storing scores for each template across time
        self._df = pd.DataFrame(np.nan, index=np.arange(self._Nt), columns=self._template_labels)

        # Dynamic map for live updates
        if streaming:
            self.dmap = hv.DynamicMap(self._plot, streams=[Stream.define('Next', t=int)()])
        else:
            self.dmap = None
        
        # Static overlays accumulated over time
        self._polys_static = []

        # Gray box before matching starts
        self._no_match_poly = self._draw_poly(
                0, self._match_start
            ).opts(color='gray', line_color=None, alpha=0.2)

        self._action_state = None
        self._last_cooldown_shown = None
        
        self._colors = self._get_template_colors()
        
        self._out_prefix = config.out_prefix
        self._out_dir = config.out_dir
        
    def update(self, t: int, data: np.ndarray, action_state: ActionState) -> None:
        """
        Update the plot with new score data at a given time point.

        Parameters
        ----------
        t : int
            TR corresponding to the incoming scores.
        data : np.ndarray
            Array of template matching scores at time `t`.
            Must align with the template label order.
        action_state : ActionState
            Current action state containing action onsets, offsets, and cooldown info.
        """
        self._df.iloc[t] = data
        self._action_state = action_state

        # Trigger redraw only if data is valid
        if not np.isnan(self._df.iloc[t]).all():
            self.dmap.event(t=t)

    def _plot(self, t: int) -> hv.Overlay:
        return self._build_overlay(t, update_polys=True)

    def _build_overlay(self, t: int, update_polys: bool) -> hv.Overlay:
        """
        Construct the full plot overlay for a given time point.

        This includes:
        - Line plots of template scores
        - Hit threshold line
        - action hit markers
        - action, cooldown, and pre-matching shaded regions

        Parameters
        ----------
        t : int
            Current time index used to update dynamic elements.

        Returns
        -------
        hv.Overlay
            Combined HoloViews overlay for rendering.
        """
        line_plot = self._df.hvplot.line(
            legend='top',
            width=1200,
            cmap=self._colors,
            group_label='Template',
            value_label='Score',
        )

        overlays = [line_plot, self._no_match_poly]

        if self._action_state is None:
            return hv.Overlay(overlays).opts(ylim=self._y_range)

        # Threshold line
        overlays.append(
            hv.HLine(self._hit_thr).opts(
                color='black', line_dash='dashed', line_width=1
            )
        )

        # Hit markers
        overlays.append(self._draw_hit_markers())

        if update_polys:
            # Action state-dependent dynamic shaded regions
            if self._action_state.in_action:
                overlays.append(self._draw_dynamic_box(t))
            elif self._action_state.action_offsets and t == self._action_state.action_offsets[-1]:
                # Final action box
                self._polys_static.append(
                        self._draw_poly(self._action_state.action_onsets[-1], t
                    ).opts(alpha=0.2, color='blue', line_color=None))
            elif self._action_state.in_cooldown:
                # Cooldown box
                if self._action_state.cooldown_end != self._last_cooldown_shown:
                    self._polys_static.append(
                        self._draw_poly(self._action_state.action_offsets[-1], self._action_state.cooldown_end)
                        .opts(alpha=0.2, color='cyan', line_color=None)
                    )
                    self._last_cooldown_shown = self._action_state.cooldown_end
        
        overlays.append(
            hv.Overlay(self._polys_static)
            if self._polys_static else hv.Overlay([])
        )
        
        return hv.Overlay(overlays).opts(ylim=self._y_range)

    def _draw_dynamic_box(self, t: int) -> hv.Polygons:
        """
        Draw a dynamic action window box extending to the current time.

        Parameters
        ----------
        t : int
            Current time index.

        Returns
        -------
        hv.Polygons
            Polygon representing the active action window.
        """
        return self._draw_poly(self._action_state.action_onsets[-1], t).opts(alpha=0.2, color='blue', )
        
    def _draw_poly(self, start: int, end: int) -> hv.Polygons:
        """
        Draw a rectangular polygon spanning a time interval.

        Parameters
        ----------
        start : int
            Start time index.
        end : int
            End time index.

        Returns
        -------
        hv.Polygons
            Rectangle covering the interval.
        """
        end = min(end, self._Nt)
        y_min, y_max = self._y_range
        return hv.Polygons([
            [(start, y_min), (end, y_min), (end, y_max), (start, y_max)]
        ])

    def _threshold_y_range(self) -> tuple[float, float]:
        """
        Return a stable live y-range based on the hit threshold.
        """
        limit = abs(self._hit_thr) * 5.0
        return (-limit, limit)

    def _data_y_range(self) -> tuple[float, float]:
        """
        Return a final y-range based on all saved scores and the hit threshold.
        """
        scores = self._df.to_numpy(dtype=float).ravel()
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size == 0:
            return self._threshold_y_range()

        values = np.concatenate([finite_scores, np.array([0.0, self._hit_thr])])
        y_min = float(values.min())
        y_max = float(values.max())
        if y_min == y_max:
            return self._threshold_y_range()

        padding = (y_max - y_min) * 0.25
        return (y_min - padding, y_max + padding)

    def _redraw_static_polys(self, t: int) -> None:
        """
        Rebuild static action and cooldown boxes using the active y-range.
        """
        self._polys_static = []
        if self._action_state is None:
            return

        for onset, offset in zip(self._action_state.action_onsets, self._action_state.action_offsets):
            self._polys_static.append(
                self._draw_poly(onset, offset).opts(alpha=0.2, color='blue', line_color=None)
            )
            self._polys_static.append(
                self._draw_poly(offset, offset + self._vols_noaction).opts(alpha=0.2, color='cyan', line_color=None)
            )

        if self._action_state.in_action and len(self._action_state.action_onsets) > len(self._action_state.action_offsets):
            self._polys_static.append(
                self._draw_poly(self._action_state.action_onsets[-1], t).opts(alpha=0.2, color='blue', line_color=None)
            )

    def _draw_hit_markers(self) -> hv.Scatter:
        """
        Draw scatter markers at action onset (hit) times.

        Each marker is placed at the score of the highest-scoring template
        at that time point and colored according to the template identity.

        Returns
        -------
        hv.Scatter
            Scatter plot of action hit markers.
        """
        points = []
        for hit_time in self._action_state.action_onsets:
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
        """
        Assign colors to each template label.

        Colors are drawn from the Category10 palette and cycled if the number
        of templates exceeds the palette size.

        Returns
        -------
        dict
            Mapping from template label to color string.
        """
        palette = Category10[10]

        # Cycle through palette if more labels than colors
        assigned_colors = [
            c for _, c in zip(self._template_labels, cycle(palette))
        ]
        return dict(zip(self._template_labels, assigned_colors))
    
    def close(self):
        """
        Save the final state of the plot to an HTML file.
        """
        out_html = osp.join(self._out_dir, self._out_prefix + '.dyn_report')
        renderer = hv.renderer('bokeh')

        # Get last time index with valid data
        valid_scores = self._df.dropna(how='all')
        last_valid_idx = valid_scores.index.max() if not valid_scores.empty else 0
        self._y_range = self._data_y_range()
        self._no_match_poly = self._draw_poly(0, self._match_start).opts(color='gray', line_color=None, alpha=0.2)
        self._redraw_static_polys(last_valid_idx)
        final_plot = self._build_overlay(last_valid_idx, update_polys=False)

        renderer.save(final_plot, out_html)
        print(f'++ Score report written to disk: [{out_html}.html]')
    
    def render_static(self, df: pd.DataFrame, action_state: ActionState) -> hv.Overlay:
        """
        Render a static score plot after the experiment has completed.

        Parameters
        ----------
        df : pd.DataFrame
            Full score DataFrame indexed by time and template.
        action_state : ActionState
            Final action state containing all action intervals.

        Returns
        -------
        hv.Overlay
            Rendered plot overlay.
        """
        self._df = df
        self._action_state = action_state
        self.close()
