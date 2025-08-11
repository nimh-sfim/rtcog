"""
Main entry point for the rtcog pipeline.

This script starts the communication process responsible for receiving scanner data
and sets up a controller for responding to experiment state if given.
"""

import sys
import os.path as osp
import multiprocessing as mp

from rtfmri.utils.options import Options
from rtfmri.utils.log import get_logger, set_logger
from rtfmri.utils.core import SharedClock, create_sync_events
from rtfmri.comm.comm_process import comm_process
from rtfmri.controller.controller import Controller
from rtfmri.controller.action_series import LatencyTestActionSeries, ESAMActionSeries
from rtfmri.experiment_registry import EXPERIMENT_REGISTRY

log = get_logger()

def main():
    """
    Main control flow for the rtfMRI experiment.

    - Parses command-line arguments and YAML config
    - Sets up shared memory and synchronization
    - Launches the communication subprocess
    - Optionally creates a controller for experiment interaction
    """
    # 1) Read Input Parameters: port, fullscreen, etc..
    # ------------------------------------------
    opts = Options.from_cli()
    opts.save_config()
    
    print(opts)

    set_logger(debug=opts.debug, silent=opts.silent)
    
    sync = create_sync_events()
        
    if opts.exp_type not in EXPERIMENT_REGISTRY:
        raise ValueError(f"Unsupported experiment type: {opts.exp_type}")

    exp_class = EXPERIMENT_REGISTRY[opts.exp_type]["processor"]
    action_class = EXPERIMENT_REGISTRY.get(opts.exp_type, {}).get("action", None)
    
    controller = None
    shared_responses = None
    clock = None
    receiver_path = None

    if action_class and not opts.no_action:
        if opts.test_latency:
            receiver_path = osp.join(opts.out_dir, f'{opts.out_prefix}_receiver_timing.pkl')
            clock = SharedClock()
            action_class = LatencyTestActionSeries
            action = action_class(sync, opts, clock=clock, receiver_path=receiver_path)
        else:
            action = action_class(sync, opts)
        controller = Controller(sync, action)

        if isinstance(action, ESAMActionSeries):
            shared_responses = action.gui.shared_responses

    # 2) Start communication process
    # ------------------------------------------
    comm_proc = mp.Process(target=comm_process, args=(opts, sync, exp_class, shared_responses, clock, receiver_path))
    comm_proc.start()

    # 3) Run controller
    # ------------------------------------
    if controller:
        controller.run()
    
    # 4) Wait for communication process to complete
    # ------------------------------------
    comm_proc.join()


if __name__ == "__main__":
    sys.exit(main())
