import time

from rtcog.utils.log import get_logger
from rtcog.comm.receiver_interface import CustomReceiverInterface
from rtcog.processor.esam_processor import ESAMProcessor

log = get_logger()

def comm_process(opts, sync, proc_class, shared_responses=None, clock=None, time_path=None, minimal=False):
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
    log.info('1) Initializing Processor...')
    processor = proc_class(opts, sync, minimal=minimal)
    if isinstance(processor, ESAMProcessor) and not minimal:
        processor.start_streaming(shared_responses) # Start streaming process

    log.info('2) Opening Communication Channel...')

    auto_save = opts.auto_save if hasattr(opts, "auto_save") else False
    receiver = CustomReceiverInterface(port=opts.tcp_port, show_data=opts.show_data, auto_save=auto_save, clock=clock, out_path=time_path)
    if not receiver:
        return 1

    if not receiver.RTI:
        log.error('RTI is not initialized.')

    log.info('3) Setting Signal Handlers...')
    receiver.set_signal_handlers()

    # Set receiver callback
    receiver.compute_TR_data = processor.compute_TR_data
    receiver.final_steps     = processor.end_run

    # Prepare for incoming connections
    log.info('4) Prepare for Incoming Connections...')
    if receiver.RTI.open_incoming_socket():
        return 1
    
    # Run experiment
    log.info('5) Ready to go...')
    rv = receiver.process_one_run()

    if opts.test_latency:
        receiver.save_timing()

    if isinstance(processor, ESAMProcessor) and not opts.no_action and not minimal:
        if sync.hit.is_set():
            log.info('Waiting for action to end')
        while sync.hit.is_set():
            time.sleep(1)
    log.info('Ready to end')

    return rv
