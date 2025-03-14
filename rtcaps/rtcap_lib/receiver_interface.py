# Rebuilding the ReceiverInterface class using updated RTInterface class (imported from abin)

import sys
import os.path as osp
import shutil
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.ERROR)

afni_path = shutil.which('afni')

if not afni_path:
    print('++ ERROR: AFNI not found in the system PATH')
    sys.exit(-1)

abin_path = osp.dirname(afni_path)
sys.path.insert(1, abin_path)

# system libraries : test, then import as local symbols
from afnipy import module_test_lib
testlibs = ['signal', 'time']
if module_test_lib.num_import_failures(testlibs): sys.exit(1)
import signal, time

# AFNI libraries (besides module_test_lib)
from afnipy import lib_realtime as RT
from afnipy import afni_util as UTIL

class ReceiverInterface:
    def __init__(self, port=None, show_data=False, verb=1):
        self.RTI = RT.RTInterface()
        
        if port:
            self.RTI.server_port = port

        # # All data (light)
        # self.RTI.version = 3
        
        if not self.RTI:
            return

        # Show Some part of the data on every TR
        self.RTI.show_data = show_data

        self.verb = verb

        # callbacks
        self.compute_TR_data = None
        self.final_steps     = None

        self.TR_data         = []

    def __del__(self):

        self.close_data_ports()

    def close_data_ports(self):
        """close TCP and socket ports, except for server port"""
        if self.RTI:
            self.RTI.close_data_ports()
    
    def set_signal_handlers(self):
        """capture common termination signals, to properly close ports"""

        if self.verb > 1: print('++ setting signals')

        slist = [ signal.SIGHUP, signal.SIGINT, signal.SIGQUIT, signal.SIGTERM ]
        if self.verb > 2: print('   signals are %s' % slist)

        for sig in slist: signal.signal(sig, self.clean_n_exit)

        return
  
    def process_one_TR(self):
        """return 0 to continue, 1 on valid termination, -1 on error"""

        log.debug("++ Entering process_one_TR()")

        # RTI.read_TR_data() used to return a tuple, now returns one value denoting if
        # program ran successfully (rv=0)
        rv = self.RTI.read_TR_data()
        if rv:
            print('** process 1 TR: read data failure')
            return rv
        
        print(f'++ DEBUG: motion = {self.RTI.motion}')
        print(f'++ DEBUG: extras = {self.RTI.extras}')
        #log.debug("Motion: %s, Extra: %s" % (motion, extra))
        # if callback is registered
        data = None
        if self.compute_TR_data:
            data = self.compute_TR_data(self.RTI.motion, self.RTI.extras)  # PROCESS DATA HERE
       
        print(f'++ DEBUG: data = {data}')

        if not data:
            return 1

        return 0

    def process_one_run(self):

        """repeatedly: process all incoming data for a single run
            return  0 on success and 1 on error
        """

        #log.info("++ Entering process_one_run()")
        print("++ Entering process_one_run()")
        # wait for the real-time plugin to talk to us
        if self.RTI.wait_for_new_run():
            return 1

        # process one TR at a time until
        #log.info('-- incoming data')
        print('-- incoming data')

        rv = self.process_one_TR()
        while rv == 0:
            rv = self.process_one_TR()
        #log.info("++ The life of this program is coming to an end....")
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

    def clean_n_exit(self, signum, frame):

      if self.verb > 1: print('++ signal handler called with signal', signum)

      self.RTI.close_data_ports()
   
      try: sys.stdout.flush()
      except: pass

      # at last, close server port
      if self.RTI.server_sock:
         if self.verb > 1: print('closing server port...')
         try: self.RTI.server_sock.close()
         except (RT.socket.error, RT.socket.timeout): pass

      if self.verb > 0: print('-- exiting on signal %d...' % signum)
      sys.exit(signum)