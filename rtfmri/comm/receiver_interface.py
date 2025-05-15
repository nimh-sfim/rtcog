# Updating ReceiverInterface class to suit our purposes

import sys
import os.path as osp
import shutil

# from config import setup_logger
import logging
log = logging.getLogger('receiver')

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
    def __init__(self, port=None, show_data=False, verb=1):
        super().__init__()
        self.RTI = RT.RTInterface()
        
        if port:
            self.RTI.server_port = port
        
        if not self.RTI:
            return
        
        # Show Some part of the data on every TR
        self.RTI.show_data = show_data

        self.verb = verb

        # callbacks
        self.compute_TR_data = None
        self.final_steps     = None

        self.TR_data         = []

    def process_one_TR(self):
        """return 0 to continue, 1 on valid termination, -1 on error"""
        # TODO: clear old data 
        # del self.RTI.extras, self.RTI.motion
        # will have to update this_tr_data logic...

        log.debug("++ Entering process_one_TR()")

        rv = self.RTI.read_TR_data()
        if rv:
            log.error('** process 1 TR: read data failure')
            return rv
        log.debug('len(motion): ' + str(len(self.RTI.motion)))
        log.debug('len(extras): ' + str(len(self.RTI.extras)))
        log.debug(self.RTI.extras[0:10])

        # if callback is registered
        data = None
        if self.compute_TR_data:
            data = self.compute_TR_data(self.RTI.motion, self.RTI.extras)  # PROCESS DATA HERE
       
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

        rv = self.process_one_TR()
        while rv == 0:
            rv = self.process_one_TR()

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
