from unittest.mock import MagicMock, patch
from rtcog.controller.controller import Controller


@patch('time.sleep')
def test_controller_run(mock_sleep):
    sync = MagicMock()
    action_series = MagicMock()

    sync.end.is_set.side_effect = [False, False, True]  # End after 2 loops
    sync.hit.is_set.side_effect = [False, True, False]  # Hit on second loop

    controller = Controller(sync, action_series)
    controller.run()

    action_series.on_start.assert_called_once()

    assert action_series.on_loop.call_count == 2

    action_series.on_hit.assert_called_once()
    action_series.on_end.assert_called_once()

    assert mock_sleep.call_count == 2


@patch('time.sleep')
def test_controller_run_no_hit(mock_sleep):
    sync = MagicMock()
    action_series = MagicMock()

    sync.end.is_set.side_effect = [False, True]
    sync.hit.is_set.return_value = False

    controller = Controller(sync, action_series)
    controller.run()

    action_series.on_start.assert_called_once()
    assert action_series.on_loop.call_count == 1
    action_series.on_hit.assert_not_called()
    action_series.on_end.assert_called_once()
    assert mock_sleep.call_count == 1


@patch('time.sleep')
def test_controller_run_immediate_end(mock_sleep):
    sync = MagicMock()
    action_series = MagicMock()

    sync.end.is_set.return_value = True
    sync.hit.is_set.return_value = False

    controller = Controller(sync, action_series)
    controller.run()

    action_series.on_start.assert_called_once()
    action_series.on_loop.assert_not_called()
    action_series.on_hit.assert_not_called()
    action_series.on_end.assert_called_once()
    mock_sleep.assert_not_called()