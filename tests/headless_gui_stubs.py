import sys
import types


def install_headless_gui_stubs():
    psychopy = types.ModuleType("psychopy")
    psychopy_event = types.ModuleType("psychopy.event")
    psychopy_event.getKeys = lambda keys=None: []
    psychopy.event = psychopy_event

    gui_utils = types.ModuleType("rtcog.gui.gui_utils")
    gui_utils.validate_likert_questions = lambda q_path: []

    base_gui = types.ModuleType("rtcog.gui.base_gui")
    base_gui.BaseGUI = object

    class BasicGUI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def draw_resting_screen(self):
            pass

        def close_psychopy_window(self):
            pass

        def poll_trigger(self):
            pass

        def save_trigger(self):
            pass

    basic_gui = types.ModuleType("rtcog.gui.basic_gui")
    basic_gui.BasicGUI = BasicGUI

    class EsamGUI(BasicGUI):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.responses = {}

        def run_full_action(self):
            return {}

        def save_likert_files(self):
            pass

    esam_gui = types.ModuleType("rtcog.gui.esam_gui")
    esam_gui.EsamGUI = EsamGUI

    sys.modules.setdefault("psychopy", psychopy)
    sys.modules.setdefault("psychopy.event", psychopy_event)
    sys.modules.setdefault("rtcog.gui.gui_utils", gui_utils)
    sys.modules.setdefault("rtcog.gui.base_gui", base_gui)
    sys.modules.setdefault("rtcog.gui.basic_gui", basic_gui)
    sys.modules.setdefault("rtcog.gui.esam_gui", esam_gui)
