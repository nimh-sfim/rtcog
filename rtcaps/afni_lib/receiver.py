import sys
import signal
from . import realtime as rt
import platform

import logging
log = logging.getLogger(__name__)

class ReceiverInterface(object):

   """User-friendly interface to AFNI real-time receiver."""

   def __init__(self, port=None, show_data=False, verb=1):

      # Real-Time interface RTInterface
      if port:
         self.RTI = rt.RTInterface(port=port)
      else:
         self.RTI = rt.RTInterface()
      if not self.RTI:
         return None

      # Show Some part of the data on every TR
      self.RTI.show_data = show_data

      self.verb = verb

      # callbacks
      self.compute_TR_data = None
      self.final_steps     = None

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

      motion, extra = self.RTI.read_TR_data()
      if not motion:
         log.error('** process 1 TR: read data failure')
         return 1

      #log.debug("Motion: %s, Extra: %s" % (motion, extra))
      # if callback is registered
      data = None
      if self.compute_TR_data:
         data = self.compute_TR_data(motion, extra)  # PROCESS DATA HERE

      log.debug("Result: %s" % data)

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
         except (rt.socket.error, rt.socket.timeout): pass

      if self.verb > 0: print('-- exiting on signal %d...' % signum)
      sys.exit(signum)
