import time

from rtfmri.utils.log import get_logger
from rtfmri.comm.receiver_interface import CustomReceiverInterface
from rtfmri.core.experiment import Experiment, ESAMExperiment

log = get_logger()

def comm_process(opts, sync, shared_responses=None, clock=None, time_path=None):
    log.info('2) Instantiating Experiment Object...')
    if opts.exp_type == 'esam':
        log.info('This an experimental run')
        experiment = ESAMExperiment(opts, sync)
        experiment.start_streaming(shared_responses) # Start panel server
        # TODO: add event to signal when server is ready before printing ready to go
    else:
        experiment = Experiment(opts, sync)

    # 4) Start Communications
    log.info('3) Opening Communication Channel...')

    auto_save = opts.auto_save if hasattr(opts, "auto_save") else False
    receiver = CustomReceiverInterface(port=opts.tcp_port, show_data=opts.show_data, auto_save=auto_save, clock=clock, out_path=time_path)
    if not receiver:
        return 1

    if not receiver.RTI:
        log.error('RTI is not initialized.')
    else:
        log.debug('RTI initialized successfully.')

    if not receiver:
        return 1

    # 5) set signal handlers and look for data
    log.info('4) Setting Signal Handlers...')
    receiver.set_signal_handlers()

    # 6) set receiver callback
    receiver.compute_TR_data  = experiment.compute_TR_data
    receiver.final_steps      = experiment.end_run

    # 7) prepare for incoming connections
    log.info('5) Prepare for Incoming Connections...')
    if receiver.RTI.open_incoming_socket():
        return 1
    
    # 8) Vinai's alternative
    log.info('6) Here we go...')
    rv = receiver.process_one_run()
    
    if opts.test_latency:
        receiver.save_timing()

    if experiment.exp_type == "esam" and not opts.no_gui:
        while experiment.sync.hit.is_set():
            log.info('waiting for QA to end ')
            time.sleep(1)
    log.info('ready to end ')
    return rv
