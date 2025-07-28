import sys
import os.path as osp
import logging
import multiprocessing as mp
import time

from rtfmri.utils.options import Options
from rtfmri.utils.log import get_logger, set_logger
from rtfmri.utils.gui import validate_likert_questions, get_experiment_info
from rtfmri.utils.core import SharedClock, create_sync_events, run_gui

from rtfmri.utils.log import get_logger

log = get_logger()

def main():
    # 1) Read Input Parameters: port, fullscreen, etc..
    # ------------------------------------------
    opts = Options.from_cli()
    opts.save_config()
    
    print(opts)

    log = set_logger(debug=opts.debug, silent=opts.silent)
        
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


def comm_process(opts, sync, shared_responses=None, clock=None, time_path=None):
    from rtfmri.comm.receiver_interface import CustomReceiverInterface
    from rtfmri.core.experiment import Experiment, ESAMExperiment
    
    # 2) Create Experiment Object
    log.info('- comm_process - 2) Instantiating Experiment Object...')
    if opts.exp_type == 'esam':
        log.info('This an experimental run')
        experiment = ESAMExperiment(opts, sync)
        experiment.start_streaming(shared_responses) # Start panel server
        # TODO: add event to signal when server is ready before printing ready to go
    else:
        experiment = Experiment(opts, sync)

    # 4) Start Communications
    log.info('- comm_process - 3) Opening Communication Channel...')

    auto_save = opts.auto_save if hasattr(opts, "auto_save") else False
    receiver = CustomReceiverInterface(port=opts.tcp_port, show_data=opts.show_data, auto_save=auto_save, clock=clock, out_path=time_path)
    if not receiver:
        return 1

    if not receiver.RTI:
        log.error('comm_process - RTI is not initialized.')
    else:
        log.debug('comm_process - RTI initialized successfully.')

    if not receiver:
        return 1

    # 5) set signal handlers and look for data
    log.info('- comm_process - 4) Setting Signal Handlers...')
    receiver.set_signal_handlers()

    # 6) set receiver callback
    receiver.compute_TR_data  = experiment.compute_TR_data
    receiver.final_steps      = experiment.end_run

    # 7) prepare for incoming connections
    log.info('- comm_process - 5) Prepare for Incoming Connections...')
    if receiver.RTI.open_incoming_socket():
        return 1
    
    # 8) Vinai's alternative
    log.info('6) Here we go...')
    rv = receiver.process_one_run()
    
    if opts.test_latency:
        receiver.save_timing()

    if experiment.exp_type == "esam" or experiment.exp_type == "esam_test":
        while experiment.sync.hit.is_set():
            log.info('- comm_process - waiting for QA to end ')
            time.sleep(1)
    log.info('- comm_process - ready to end ')
    return rv

if __name__ == "__main__":
    sys.exit(main())
