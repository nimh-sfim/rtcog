import sys
import argparse

from rtfmri.utils.gui import DefaultGUI, get_experiment_info
from rtfmri.utils.core import SharedClock

def get_opts(options=None):
    parser = argparse.ArgumentParser(
        description="Make sure psychopy is detecting keystrokes"
    )

    parser_save   = parser.add_argument_group("Saving Options")
    parser_save.add_argument("--out_dir", help="Output directory  [default: %(default)s]", dest="out_dir", action="store", type=str, default="./")
    parser_save.add_argument("--out_prefix",  help="Prefix for outputs", dest="out_prefix", action="store", type=str)
    
    return parser.parse_args(options)

def main():
    opts = get_opts(sys.argv[1:])
    print(opts)
    exp_info = get_experiment_info(opts)

    clock = SharedClock()

    gui = DefaultGUI(exp_info, opts, clock)
    
    try:
        while True:
            gui.draw_resting_screen()
            gui.poll_trigger()
    except KeyboardInterrupt:
        gui.save_trigger()
        gui.close_psychopy_infrastructure()
    
if __name__ == "__main__":
    main()