import pytest
from unittest.mock import MagicMock, patch
from rtcog.controller.action_series import BasicActionSeries, ESAMActionSeries, LatencyTestActionSeries
import multiprocessing as mp

def test_exp_ends_if_esc_key():
    sync = MagicMock()
    sync.end = mp.Event()

    gui = MagicMock()
    action = BasicActionSeries(sync=sync, opts={}, gui=gui)

    with patch('psychopy.event.getKeys', return_value=['escape']):
        action.on_loop()

    assert sync.end.is_set()

@patch('rtcog.controller.action_series.EsamGUI')
@patch('rtcog.controller.action_series.validate_likert_questions')
def test_esam_on_hit(mock_validate, mock_esam_gui):
    mock_validate.return_value = [{"name": "q1"}]

    gui = MagicMock()
    gui.run_full_action.return_value = {"q1": ("agree", 1.0)}
    mock_esam_gui.return_value = gui

    sync = MagicMock()
    sync.hit = mp.Event()
    sync.hit.set()
    sync.action_end = mp.Event()

    mock_opts = MagicMock()
    mock_opts.q_path = "fake/path"

    action = ESAMActionSeries(sync=sync, opts=mock_opts)

    action.on_hit()

    gui.run_full_action.assert_called_once()
    assert not sync.hit.is_set()
    assert sync.action_end.is_set()

@patch('rtcog.controller.action_series.BasicGUI')
def test_latencytest_on_loop_calls_poll_trigger(mock_basic_gui):
    sync = MagicMock()
    sync.end = mp.Event()

    opts = {}
    clock = MagicMock()

    action = LatencyTestActionSeries(sync=sync, opts=opts, clock=clock)
    action.gui = MagicMock()

    action.on_loop(),
    action.gui.poll_trigger.assert_called_once()


if __name__ == "__main__":
    pytest.main()