import sys
import os.path as osp
import logging
import multiprocessing as mp
import time

from rtfmri.utils.options import Options
from rtfmri.utils.log import get_logger, set_logger
from rtfmri.utils.gui import validate_likert_questions, get_experiment_info
from rtfmri.utils.core import SharedClock, create_sync_events, run_gui
from rtfmri.comm.comm_process import comm_process

from rtfmri.utils.log import get_logger

log = get_logger()

def main():
    # 1) Read Input Parameters: port, fullscreen, etc..
    # ------------------------------------------
    opts = Options.from_cli()
    opts.save_config()
    
    print(opts)

    set_logger(debug=opts.debug, silent=opts.silent)
        
    shared_responses = None
    clock = None
    receiver_path = None
    if opts.test_latency:
        clock = SharedClock()
        receiver_path = osp.join(opts.out_dir, f'{opts.out_prefix}_receiver_timing.pkl')
    if opts.exp_type == "esam":
        opts.likert_questions = validate_likert_questions(opts.q_path)
        manager = mp.Manager()
        shared_responses = manager.dict({q["name"]: (None, None) for q in opts.likert_questions})

    # 2) Create Multi-processing infrastructure
    # ------------------------------------------
    sync = create_sync_events()
    
    comm_proc = mp.Process(target=comm_process, args=(opts, sync, shared_responses, clock, receiver_path))
    comm_proc.start()

    # 3) Get additional info using the GUI
    # ------------------------------------
    if not opts.no_gui:
        exp_info = get_experiment_info(opts)
        run_gui(opts, exp_info, sync, clock, shared_responses)
    else:
        while not sync.end.is_set():
            time.sleep(0.1)
    
    comm_proc.join()




if __name__ == "__main__":
    sys.exit(main())
