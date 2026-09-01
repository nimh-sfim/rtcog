import pytest
from unittest.mock import MagicMock, patch
import multiprocessing as mp
from types import SimpleNamespace

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


@patch('rtcog.controller.action_series.pd.read_pickle')
def test_latency_metrics_reads_files_from_output_directory(mock_read_pickle, tmp_path):
    mock_read_pickle.side_effect = [
        [0.0, 1.0],
        {"recv": [0.1, 1.1], "proc": [0.2, 1.2]},
    ]
    action = LatencyTestActionSeries.__new__(LatencyTestActionSeries)
    action.opts = SimpleNamespace(
        out_dir=str(tmp_path),
        out_prefix="run",
        nvols=2,
        discard=0,
    )

    result = action._calculate_latency_metrics()

    assert result is not None
    assert [call.args[0] for call in mock_read_pickle.call_args_list] == [
        str(tmp_path / "run_trigger_timing.pkl"),
        str(tmp_path / "run_receiver_timing.pkl"),
    ]


if __name__ == "__main__":
    pytest.main()
