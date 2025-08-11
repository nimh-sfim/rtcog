# Updating ReceiverInterface class to suit our purposes

import sys
import os.path as osp
import shutil
import traceback
import pickle

sys.path.append('..')
from rtfmri.utils.exceptions import VolumeOverflowError

import logging
log = logging.getLogger(__name__)

afni_path = shutil.which('afni')

if not afni_path:
    log.error('++ ERROR: AFNI not found in the system PATH')
    sys.exit(-1)

abin_path = osp.dirname(afni_path)
sys.path.insert(1, abin_path)

# system libraries : test, then import as local symbols
from afnipy import module_test_lib
testlibs = ['signal', 'time']
if module_test_lib.num_import_failures(testlibs): sys.exit(1)

# AFNI libraries (besides module_test_lib)
from realtime_receiver import ReceiverInterface
from afnipy import lib_realtime as RT

    
class CustomReceiverInterface(ReceiverInterface):
    def __init__(self, port=None, show_data=False, verb=0, auto_save=True, clock=None, out_path=None):
        super().__init__()
        self.RTI = RT.RTInterface()
        
        if port:
            self.RTI.server_port = port
        
        self.verb = verb
        level = {
            0: logging.ERROR,
            1: logging.INFO,
            2: logging.DEBUG
        }.get(verb, logging.ERROR)
        log.setLevel(level)

        if not self.RTI:
            return
        
        # Show Some part of the data on every TR
        self.RTI.show_data = show_data

        # callbacks
        self.compute_TR_data = None
        self.final_steps     = None
        
        self.auto_save = auto_save
        
        self.clock = clock
        self.timing = {"recv": [], "proc": []}
        self.out_path = out_path

    def process_one_TR(self):
        """return 0 to continue, 1 on valid termination, -1 on error"""
        # TODO: clear old data 
        # del self.RTI.extras, self.RTI.motion
        # will have to update this_tr_data logic...

        log.debug("++ Entering process_one_TR()")

        rv = self.RTI.read_TR_data()
        if self.clock and not rv:
            recv_time = self.clock.now()
            self.timing["recv"].append(recv_time)
            print(f"Recv @ {recv_time:.3f}", flush=True)

        if rv:
            log.error('** process 1 TR: read data failure')
            return rv

        # if callback is registered
        data = None

        if self.compute_TR_data:
            data = self.compute_TR_data(self.RTI.motion, self.RTI.extras)
            if self.clock:
                proc_time = self.clock.now()
                self.timing["proc"].append(proc_time)
                print(f"Proc @ {proc_time:.3f}", flush=True)

        if not data:
            return 1

        return 0

    def process_one_run(self):

        """repeatedly: process all incoming data for a single run
            return  0 on success and 1 on error
        """

        log.info("++ Entering process_one_run()")
        # wait for the real-time plugin to talk to us
        if self.RTI.wait_for_new_run():
            return 1

        # process one TR at a time until
        log.info('-- incoming data')

        try:
            rv = self.process_one_TR()
            while rv == 0:
                rv = self.process_one_TR()
        except VolumeOverflowError:
            log.error(f'++ ERROR: Receiving more volumes from the scanner than expected.')
            log.error(f'Exiting experiment...')
            if self.final_steps:
                self.final_steps()
            return
        except Exception as e:
            log.error(f"++ ERROR: An unexpected error occurred: {e}")
            log.error(traceback.format_exc())
            if self.final_steps:
                if self.auto_save:
                    self.final_steps()
                else:
                    self.final_steps(save=False)
            return

        log.info("-- The life of this program is coming to an end....")
        log.info("-- Calling the Final Steps Function...")
        if self.final_steps:
            self.final_steps()
         
        if rv > 0:
            tail = '(terminating on success)'
        else:
            tail = '(terminating on error)'
        log.info('-- processed %d TRs of data %s' % (self.RTI.nread, tail))
        log.info('-' * 60)
        
        return rv
    
    def save_timing(self):
        with open(self.out_path, 'wb') as f:
            pickle.dump(self.timing, f)
        log.info(f'Timing saved to {self.out_path}')
        
class MinimalReceiverInterface(CustomReceiverInterface):
    """Receiver interface without any data computation. Used for latency testing."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timing = {"recv": []}
        self.t = 1

    def process_one_TR(self):
        """Overridden to only track recv time, not proc time."""
        log.debug("++ Entering MinimalReceiverInterface.process_one_TR()")

        rv = self.RTI.read_TR_data()
        if self.clock and not rv:
            recv_time = self.clock.now()
            self.timing["recv"].append(recv_time)
            print(f"Recv @ {recv_time:.3f}                 - Time point [t={self.t}]", flush=True)
        self.t += 1

        if rv:
            log.error('** process 1 TR: read data failure')
            return rv

        return 0  # Always continue; no data computation