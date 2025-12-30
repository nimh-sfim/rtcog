from rtcog.viz.esam_streaming import ESAMStreamer
from types import SimpleNamespace

def test_in_cooldown():
    streamer = ESAMStreamer.__new__(ESAMStreamer)
    streamer._action_offsets = [5]
    streamer._vols_noaction = 3

    streamer._last_t = 6
    assert streamer.in_cooldown is True

    streamer._last_t = 9
    assert streamer.in_cooldown is False

def test_in_cooldown_no_offsets():
    streamer = ESAMStreamer.__new__(ESAMStreamer)
    streamer._action_onsets = [1]
    streamer._action_offsets = []
    streamer._last_t = 2
    assert streamer.in_cooldown is False

def test_update_action_state():
    action_onsets = [2]
    action_offsets = [5]

    streamer = ESAMStreamer.__new__(ESAMStreamer)
    streamer._action_onsets = action_onsets
    streamer._action_offsets = action_offsets
    streamer._vols_noaction = 2
    streamer._last_t = -1
    streamer._in_action = False
    streamer._cooldown_end = None
    streamer._hit = False

    streamer._update_action_state(2)
    assert streamer._hit is True
    assert streamer._in_action is True

    streamer._update_action_state(5)
    assert streamer._in_action is False
    assert streamer._cooldown_end == 7

def test_shutdown_calls_cleanup():
    streamer = ESAMStreamer.__new__(ESAMStreamer)
    
    class DummyServer:
        stopped = False
        def stop(self): self.stopped = True
    
    class DummyPlotter:
        closed = False
        def close(self): self.closed = True

    streamer._server = DummyServer()
    streamer._plotters = [DummyPlotter()]
    streamer._match_scores = SimpleNamespace(cleanup=lambda: setattr(streamer, "_match_scores_cleaned", True))
    streamer._tr_data = SimpleNamespace(cleanup=lambda: setattr(streamer, "_tr_data_cleaned", True))
    
    streamer._shutdown()
    assert streamer._server.stopped
    assert streamer._plotters[0].closed
    assert streamer._match_scores_cleaned
    assert streamer._tr_data_cleaned
