"""
Main entry point for the rtfMRI pipeline.

This script initializes configuration options, sets up inter-process
communication, launches a GUI if enabled, and starts the communication
process responsible for receiving scanner data or simulation input.
"""

import sys
import os.path as osp
import multiprocessing as mp
import time

from rtfmri.utils.options import Options
from rtfmri.utils.log import get_logger, set_logger
from rtfmri.gui.gui_utils import get_experiment_info, validate_likert_questions
from rtfmri.utils.core import SharedClock, create_sync_events
from rtfmri.comm.comm_process import comm_process
from rtfmri.experiment_registry import EXPERIMENT_REGISTRY

log = get_logger()

def main():
    """
    Main control flow for the rtfMRI experiment.

    - Parses command-line arguments and YAML config
    - Sets up shared memory and synchronization
    - Launches the communication subprocess
    - Optionally opens a PsychoPy GUI for experiment interaction
    """
    # 1) Read Input Parameters: port, fullscreen, etc..
    # ------------------------------------------
    opts = Options.from_cli()
    opts.save_config()
    
    print(opts)

    set_logger(debug=opts.debug, silent=opts.silent)
    
    sync = create_sync_events()
        
    shared_responses = None
    clock = SharedClock() if opts.test_latency else None
    receiver_path = osp.join(opts.out_dir, f'{opts.out_prefix}_receiver_timing.pkl') if opts.test_latency else None
    
    if opts.exp_type not in EXPERIMENT_REGISTRY:
        raise ValueError(f"Unsupported experiment type: {opts.exp_type}")

    exp_class = EXPERIMENT_REGISTRY[opts.exp_type]["experiment"]
    gui_class = EXPERIMENT_REGISTRY[opts.exp_type]["gui"]
    action_class = EXPERIMENT_REGISTRY[opts.exp_type]["action"]
    
    # TODO: move this out of main
    if opts.exp_type == "esam":
        opts.likert_questions = validate_likert_questions(opts.q_path)
        manager = mp.Manager()
        shared_responses = manager.dict({q["name"]: (None, None) for q in opts.likert_questions})

    gui = None
    if not opts.no_gui:
        exp_info = get_experiment_info(opts)
        gui_kwargs = {
            "expInfo": exp_info,
            "opts": opts,
            "clock": clock,
            "shared_responses": shared_responses,
        }
        gui = gui_class(**gui_kwargs)


    action = action_class(sync, gui)

    # 2) Start communication process
    # ------------------------------------------
    comm_proc = mp.Process(target=comm_process, args=(opts, sync, exp_class, shared_responses, clock, receiver_path))
    comm_proc.start()

    # 3) Run experiment
    # ------------------------------------
    action.run()
    
    # 4) Wait for communication process to complete
    # ------------------------------------
    comm_proc.join()


if __name__ == "__main__":
    sys.exit(main())
