import pytest
from unittest.mock import MagicMock, patch
import multiprocessing as mp

from headless_gui_stubs import install_headless_gui_stubs

install_headless_gui_stubs()

from rtcog.controller.action_series import BasicActionSeries, ESAMActionSeries, LatencyTestActionSeries


def test_exp_ends_if_esc_key(make_sync_mock):
    sync = make_sync_mock(end=mp.Event())

    gui = MagicMock()
    action = BasicActionSeries(sync=sync, opts={}, gui=gui)

    with patch('rtcog.controller.action_series.event.getKeys', return_value=['escape']):
        action.on_loop()

    assert sync.end.is_set()

@patch('rtcog.controller.action_series.EsamGUI')
@patch('rtcog.controller.action_series.validate_likert_questions')
@patch('rtcog.controller.action_series.mp.Manager')
def test_esam_on_hit(mock_manager, mock_validate, mock_esam_gui, make_sync_mock):
    mock_validate.return_value = [{"name": "q1"}]
    mock_manager.return_value.dict.side_effect = dict

    gui = MagicMock()
    gui.run_full_action.return_value = {"q1": ("agree", 1.0)}
    mock_esam_gui.return_value = gui

    sync = make_sync_mock(hit=mp.Event(), action_end=mp.Event())
    sync.hit.set()

    mock_opts = MagicMock()
    mock_opts.q_path = "fake/path"

    action = ESAMActionSeries(sync=sync, opts=mock_opts)

    action.on_hit()

    gui.run_full_action.assert_called_once()
    assert not sync.hit.is_set()
    assert sync.action_end.is_set()

@patch('rtcog.controller.action_series.BasicGUI')
def test_latencytest_on_loop_calls_poll_trigger(mock_basic_gui, make_sync_mock):
    sync = make_sync_mock(end=mp.Event())

    opts = {}
    clock = MagicMock()
    gui = MagicMock()
    mock_basic_gui.return_value = gui

    action = LatencyTestActionSeries(sync=sync, opts=opts, clock=clock)

    action.on_loop(),
    gui.poll_trigger.assert_called_once()


if __name__ == "__main__":
    pytest.main()
