import time

from rtfmri.utils.log import get_logger
from rtfmri.comm.receiver_interface import CustomReceiverInterface

log = get_logger()

def comm_process(opts, sync, proc_class, shared_responses=None, clock=None, time_path=None):
    """
    Main communication process handling experiment setup and data reception.

    Parameters
    ----------
    opts : Options
        Configuration options for the experiment run.
    sync : SyncEvents
        Synchronization events object for interprocess signaling.
    proc_class : obj
        The processor class to be instantiated.
    shared_responses : DictProxy, optional
        Shared dictionary for storing participant responses (default is None).
    clock : SharedClock, optional
        Optional timing object for latency measurements (default is None).
    time_path : str, optional
        Optional path for saving timing data (default is None).

    Returns
    -------
    int
        Return code indicating success (0) or failure (1).
    """
    log.info('2) Instantiating Processor Object...')
    processor = proc_class(opts, sync)
    # TODO: move this somewhere else
    if opts.exp_type == 'esam':
        log.info('This an experimental run')
        processor.start_streaming(shared_responses) # Start panel server

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
    receiver.compute_TR_data  = processor.compute_TR_data
    receiver.final_steps      = processor.end_run

    # 7) prepare for incoming connections
    log.info('5) Prepare for Incoming Connections...')
    if receiver.RTI.open_incoming_socket():
        return 1
    
    # 8) Vinai's alternative
    log.info('6) Here we go...')
    rv = receiver.process_one_run()
    
    if opts.test_latency:
        receiver.save_timing()

    # TODO: refactor
    if processor.exp_type == "esam" and not opts.no_action:
        while processor.sync.hit.is_set():
            log.info('waiting for QA to end ')
            time.sleep(1)
    log.info('ready to end ')
    return rv
