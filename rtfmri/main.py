import sys
import os.path as osp
import logging
import multiprocessing as mp
import time

from psychopy import event
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), 'core')))

from core.options import Options
from utils.log import get_logger, set_logger
from utils.experiment_qa import  get_experiment_info, DefaultScreen, QAScreen

log = get_logger()

def main():
    # 1) Read Input Parameters: port, fullscreen, etc..
    # ------------------------------------------
    opts = Options.from_cli()
    opts.save_config()

    log = set_logger(debug=opts.debug, silent=opts.silent)

    if opts.exp_type in ['esam', 'esam_test']:
        # have function to verify files
        pass

    # 2) Create Multi-processing infrastructure
    # ------------------------------------------
    mp_evt_hit    = mp.Event() # Start with false
    mp_evt_end    = mp.Event() # Start with false
    mp_evt_qa_end = mp.Event() # Start with false
    mp_prc_comm   = mp.Process(target=comm_process, args=(opts, mp_evt_hit, mp_evt_end, mp_evt_qa_end))
    mp_prc_comm.start()

    # 3) Get additional info using the GUI
    # ------------------------------------
    if not opts.no_gui:
        exp_info = get_experiment_info(opts)

    # add logic for esam
    if opts.exp_type == "preproc":
        if not opts.no_gui:
            # 4) Start GUI
            rest_exp = DefaultScreen(exp_info, opts)

            # 5) Keep the experiment going, until it ends
            while not mp_evt_end.is_set():
                rest_exp.draw_resting_screen()
                if event.getKeys(['escape']):
                    log.info('- User pressed escape key')
                    mp_evt_end.set()

            # 6) Close Psychopy Window
            # ------------------------
            rest_exp.close_psychopy_infrastructure()
        else:
            # 4) In no_gui mode, wait passively for experiment to end
            while not mp_evt_end.is_set():
                time.sleep(0.1)
    

def comm_process(opts, mp_evt_hit, mp_evt_end, mp_evt_qa_end):
    from comm.receiver_interface import CustomReceiverInterface
    from core.experiment import Experiment
    
    # 2) Create Experiment Object
    log.info('- comm_process - 2) Instantiating Experiment Object...')
    experiment = Experiment(opts, mp_evt_hit, mp_evt_end, mp_evt_qa_end)

    # 3) Initilize GUI (if needed):
    if experiment.exp_type == "esam" or experiment.exp_type == "esam_test":
        log.info('- comm_process - 2.a) This is an experimental run')
        experiment.setup_esam_run(opts) 
    
    # 4) Start Communications
    log.info('- comm_process - 3) Opening Communication Channel...')
    receiver = CustomReceiverInterface(port=opts.tcp_port, show_data=opts.show_data)
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
    receiver.final_steps      = experiment.final_steps

    # 7) prepare for incoming connections
    log.info('- comm_process - 5) Prepare for Incoming Connections...')
    if receiver.RTI.open_incoming_socket():
        return 1
    
    # 8) Vinai's alternative
    log.info('6) Here we go...')
    rv = receiver.process_one_run()

    if experiment.exp_type == "esam" or experiment.exp_type == "esam_test":
        while experiment.mp_evt_hit.is_set():
            log.info('- comm_process - waiting for QA to end ')
            time.sleep(1)
    log.info('- comm_process - ready to end ')
    return rv

if __name__ == "__main__":
    sys.exit(main())