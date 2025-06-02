import sys
import os.path as osp
import logging
import multiprocessing as mp
import json
import time
from psychopy import event

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), 'core')))

from paths import RESOURCES_DIR
from core.options import Options
from utils.log import get_logger, set_logger
from utils.gui import  get_experiment_info, DefaultGUI, EsamGUI

from psychopy import logging
logging.console.setLevel(logging.ERROR)

log = get_logger()

def main():
    # 1) Read Input Parameters: port, fullscreen, etc..
    # ------------------------------------------
    opts = Options.from_cli()
    opts.save_config()
    
    print(opts)

    log = set_logger(debug=opts.debug, silent=opts.silent)

    if opts.exp_type == "esam":
        if not opts.q_path:
            log.error('Path to Likert questions was not provided. Program will exit.')
            sys.exit(-1)
        if not osp.isfile(opts.q_path): # If not file, assume in RESOURCES_DIR
            fname = opts.q_path + ".json" if not opts.q_path.endswith(".json") else opts.q_path 
            opts.q_path = osp.join(RESOURCES_DIR, fname)
        try:
            with open(opts.q_path, 'r') as f:
                opts.likert_questions = json.load(f)
        except json.JSONDecodeError:
            log.error(f'The question file at {opts.q_path} is not a valid JSON.')
            sys.exit(-1)
        except Exception as e:
            log.error(f'Error loading questions at {opts.q_path}: {e}')
            sys.exit(-1)
        
    # 2) Create Multi-processing infrastructure
    # ------------------------------------------
    mp_evt_hit    = mp.Event()
    mp_evt_end    = mp.Event()
    mp_evt_qa_end = mp.Event()
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
            preproc_gui = DefaultGUI(exp_info, opts)

            # 5) Keep the experiment going, until it ends
            while not mp_evt_end.is_set():
                preproc_gui.draw_resting_screen()
                if event.getKeys(['escape']):
                    log.info('- User pressed escape key')
                    mp_evt_end.set()

            # 6) Close Psychopy Window
            # ------------------------
            preproc_gui.close_psychopy_infrastructure()
        else:
            # 4) In no_gui mode, wait passively for experiment to end
            while not mp_evt_end.is_set():
                time.sleep(0.1)

    if opts.exp_type == "esam":
        # 4) Start GUI
        # ------------
        esam_gui = EsamGUI(exp_info, opts)
    
        # 5) Wait for things to happen
        # ----------------------------
        while not mp_evt_end.is_set():
            esam_gui.draw_resting_screen()
            if event.getKeys(['escape']):
                log.info('- User pressed escape key')
                mp_evt_end.set()
            if mp_evt_hit.is_set():
                responses = esam_gui.run_full_QA()
                log.info(' - Responses: %s' % str(responses))
                mp_evt_hit.clear()
                mp_evt_qa_end.set()
        
        # 6) Close Psychopy Window
        # ------------------------
        esam_gui.save_likert_files()
        esam_gui.close_psychopy_infrastructure()


def comm_process(opts, mp_evt_hit, mp_evt_end, mp_evt_qa_end):
    from comm.receiver_interface import CustomReceiverInterface
    from core.experiment import Experiment, ESAMExperiment
    
    # 2) Create Experiment Object
    log.info('- comm_process - 2) Instantiating Experiment Object...')
    if opts.exp_type == 'esam':
        log.info('This an experimental run')
        experiment = ESAMExperiment(opts, mp_evt_hit, mp_evt_end, mp_evt_qa_end)
    else:
        experiment = Experiment(opts, mp_evt_hit, mp_evt_end, mp_evt_qa_end)

    # 4) Start Communications
    log.info('- comm_process - 3) Opening Communication Channel...')

    auto_save = opts.auto_save if hasattr(opts, "auto_save") else False
    receiver = CustomReceiverInterface(port=opts.tcp_port, show_data=opts.show_data, auto_save=auto_save)
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

    if experiment.exp_type == "esam" or experiment.exp_type == "esam_test":
        while experiment.mp_evt_hit.is_set():
            log.info('- comm_process - waiting for QA to end ')
            time.sleep(1)
    log.info('- comm_process - ready to end ')
    return rv

if __name__ == "__main__":
    sys.exit(main())