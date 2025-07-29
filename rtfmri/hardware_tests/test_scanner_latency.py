# This script is for running a very minimal version of the software that does not 
# do any data processing. It just tracks the latency between scanner acquisition
# and data receipt.
# ============================================================================
import sys
import os.path as osp
import multiprocessing as mp

from rtfmri.paths import CONFIG_DIR
from rtfmri.utils.options import Options
from rtfmri.utils.log import get_logger, set_logger
from rtfmri.utils.gui import get_experiment_info
from rtfmri.utils.core import SharedClock, create_sync_events, run_gui
from rtfmri.hardware_tests.hardware_utils import get_opts
from rtfmri.comm.receiver_interface import MinimalReceiverInterface

log = get_logger()

def main():
    opts = Options.from_yaml(osp.join(CONFIG_DIR, 'minimal_latency_config.yaml'))
    opts.test_latency = True # Ensure true in case yaml was edited...
    cli_opts = get_opts(sys.argv[1:])
    opts.out_dir, opts.out_prefix = cli_opts.out_dir, cli_opts.out_prefix

    opts.save_config()
    print(opts)
    
    set_logger(debug=opts.debug, silent=opts.silent)
    
    clock = SharedClock()
    receiver_path = osp.join(opts.out_dir, f'{opts.out_prefix}_receiver_timing.pkl')
    sync = create_sync_events()
    
    comm_proc = mp.Process(target=minimal_comm_process, args=(opts, clock, receiver_path))
    comm_proc.start()
    
    exp_info = get_experiment_info(opts)
    run_gui(opts, exp_info, sync, clock)

    comm_proc.join()

def minimal_comm_process(opts, clock, receiver_path):
    print("Starting comm process...", flush=True)
    receiver = MinimalReceiverInterface(
        port=opts.tcp_port,
        show_data=opts.show_data,
        clock=clock,
        out_path=receiver_path
    )
    if not receiver:
        return 1
    
    if not receiver.RTI:
        print('++ ERROR: RTI is not initialized.', flush=True)
    
    print('Prepare for Incoming Connections...', flush=True)
    if receiver.RTI.open_incoming_socket():
        return 1
    
    print('Here we go...', flush=True)
    rv = receiver.process_one_run()
    
    receiver.save_timing()

    print('Press escape in GUI to end', flush=True)
    return rv

if __name__ == "__main__":
    sys.exit(main())