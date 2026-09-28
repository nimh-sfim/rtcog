import inspect

import numpy as np
import pandas as pd
import pytest

from rtcog.matching.matching_opts import MatchingOpts
from rtcog.utils.sync import ActionState
from rtcog.viz.score_plotter import ScorePlotter
from rtcog.viz.streaming_config import StreamingConfig


def make_plotter(hit_thr=0.5, nt=10):
    config = StreamingConfig(
        Nt=nt,
        template_labels=["template_a", "template_b"],
        hit_thr=hit_thr,
        matching_opts=MatchingOpts(
            match_method="svr",
            match_start=2,
            vols_noaction=3,
        ),
        mask_img=None,
        Nv=1,
        out_dir=".",
        out_prefix="test",
    )
    return ScorePlotter(config, streaming=False)


def polygon_xy(polygon):
    return polygon.data[0]["x"], polygon.data[0]["y"]


def test_live_y_range_is_fixed_from_hit_threshold():
    plotter = make_plotter(hit_thr=0.5)

    assert plotter._y_range == pytest.approx((-2.5, 2.5))


def test_live_y_range_stays_small_for_nmi_scale_threshold():
    plotter = make_plotter(hit_thr=0.02)

    assert plotter._y_range == pytest.approx((-0.1, 0.1))


def test_live_y_range_has_room_for_mask_scale_threshold():
    plotter = make_plotter(hit_thr=0.8)

    assert plotter._y_range == pytest.approx((-4.0, 4.0))


def test_plot_callback_accepts_only_time_argument():
    assert list(inspect.signature(ScorePlotter._plot).parameters) == ["self", "t"]


def test_live_y_range_expands_for_large_hit_threshold():
    plotter = make_plotter(hit_thr=2.0)

    assert plotter._y_range == pytest.approx((-10.0, 10.0))


def test_saved_y_range_uses_all_scores_and_threshold():
    plotter = make_plotter(hit_thr=0.5)
    plotter._df = pd.DataFrame(
        {
            "template_a": [-2.0, 0.1, np.nan],
            "template_b": [0.2, 3.0, np.nan],
        }
    )

    assert plotter._data_y_range() == pytest.approx((-3.25, 4.25))


def test_saved_y_range_falls_back_without_valid_scores():
    plotter = make_plotter(hit_thr=0.5)

    assert plotter._data_y_range() == pytest.approx(plotter._threshold_y_range())


def test_polygons_follow_active_y_range():
    plotter = make_plotter()
    plotter._y_range = (-2.0, 3.0)

    x, y = polygon_xy(plotter._draw_poly(1, 4))

    np.testing.assert_array_equal(x, np.array([1, 4, 4, 1]))
    np.testing.assert_array_equal(y, np.array([-2.0, -2.0, 3.0, 3.0]))


def test_static_polygons_are_redrawn_from_final_action_state():
    plotter = make_plotter()
    plotter._y_range = (-2.0, 3.0)
    plotter._action_state = ActionState(
        action_onsets=[2],
        action_offsets=[5],
        in_action=False,
        in_cooldown=False,
        cooldown_end=None,
        hit=False,
    )

    plotter._redraw_static_polys(t=9)

    assert len(plotter._polys_static) == 2
    action_x, action_y = polygon_xy(plotter._polys_static[0])
    cooldown_x, cooldown_y = polygon_xy(plotter._polys_static[1])
    np.testing.assert_array_equal(action_x, np.array([2, 5, 5, 2]))
    np.testing.assert_array_equal(action_y, np.array([-2.0, -2.0, 3.0, 3.0]))
    np.testing.assert_array_equal(cooldown_x, np.array([5, 8, 8, 5]))
    np.testing.assert_array_equal(cooldown_y, np.array([-2.0, -2.0, 3.0, 3.0]))
