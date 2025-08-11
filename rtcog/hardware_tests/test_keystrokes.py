import sys

from rtcog.utils.gui import DefaultGUI, get_experiment_info
from rtcog.utils.core import SharedClock
from rtcog.hardware_tests.hardware_utils import get_opts



def main():
    opts = get_opts(sys.argv[1:])
    print(opts)
    exp_info = get_experiment_info(opts)

    clock = SharedClock()

    gui = DefaultGUI(exp_info, opts, clock)
    
    print('Press `t` to test keypresses, `escape` to end')
    while True:
        gui.draw_resting_screen()
        gui.poll_trigger()
    
if __name__ == "__main__":
    main()