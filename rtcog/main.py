"""
Main entry point for the rtcog pipeline.

This script starts the communication process responsible for receiving scanner data
and sets up a controller for responding to experiment state if given.
"""

import sys
import time
import os.path as osp
import multiprocessing as mp

from rtcog.utils.options import Options
from rtcog.utils.log import get_logger, set_logger
from rtcog.utils.core import SharedClock, create_sync_events
from rtcog.comm.comm_process import comm_process
from rtcog.controller.controller import Controller
from rtcog.controller.action_series import LatencyTestActionSeries, ESAMActionSeries
from rtcog.experiment_registry import EXPERIMENT_REGISTRY

log = get_logger()

def main():
    """
    Main control flow for the rtcog experiment.

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

    proc_class = EXPERIMENT_REGISTRY[opts.exp_type]["processor"]
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
            action = action_class(sync, opts, clock=clock)
        else:
            action = action_class(sync, opts)
        controller = Controller(sync, action)

        if isinstance(action, ESAMActionSeries):
            shared_responses = action.gui.shared_responses

    comm_proc = mp.Process(
        target=comm_process,
        args=(opts, sync, proc_class, shared_responses, clock, receiver_path)
    )

    try:
        # 1) Start communication process
        # ------------------------------------------
        comm_proc.start()

        # 2) Run controller
        # ------------------------------------
        if controller:
            controller.run()
        
        # 3) Wait for communication process to complete
        # ------------------------------------
        comm_proc.join()

    except KeyboardInterrupt:
        log.info("User interrupted, shutting down.")
    finally:
        sync.end.set()
        comm_proc.join()


if __name__ == "__main__":
    sys.exit(main())
