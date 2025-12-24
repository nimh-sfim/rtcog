"""
Minimal entry point for rtcog without GUI dependencies.
"""

import sys
import multiprocessing as mp

from rtcog.utils.options import Options
from rtcog.utils.log import set_logger
from rtcog.utils.core import SharedClock, create_sync_events
from rtcog.comm.comm_process import comm_process
from rtcog.processor.basic_processor import BasicProcessor
from rtcog.processor.esam_processor import ESAMProcessor

def main():
    opts = Options.from_cli()
    opts.save_config()

    set_logger(debug=opts.debug, silent=opts.silent)
    sync = create_sync_events()

    if opts.exp_type == "basic":
        proc_class = BasicProcessor
    elif opts.exp_type == "esam":
        proc_class = ESAMProcessor
    else:
        raise ValueError(f"Unsupported experiment type for minimal mode: {opts.exp_type}.")

    clock = SharedClock() if opts.test_latency else None
    receiver_path = opts.out_dir + f"{opts.out_prefix}_receiver_timing.pkl" if opts.test_latency else None

    comm_proc = mp.Process(
        target=comm_process,
        args=(opts, sync, proc_class, None, clock, receiver_path, True)
    )
    comm_proc.start()
    comm_proc.join()

if __name__ == "__main__":
    sys.exit(main())
