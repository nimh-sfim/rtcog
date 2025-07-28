import time
import os.path as osp
import math
from multiprocessing import Event, Value
from ctypes import c_int

from rtfmri.utils.sync import SyncEvents
from rtfmri.utils.gui import DefaultGUI, EsamGUI
from rtfmri.utils.log import get_logger

from psychopy import logging
logging.console.setLevel(logging.ERROR)

log = get_logger()

def file_exists(path):
      if not osp.isfile(path):
         raise FileNotFoundError(f"File not found: {path}")
      return path

def create_sync_events():
    """Create multiprocessing infrastructure"""
    return SyncEvents(
        new_tr=Event(),
        shm_ready=Event(),
        qa_end=Event(),
        hit=Event(),
        end=Event(),
        tr_index=Value(c_int, -1)
    )
    
def run_gui(opts, exp_info, sync, clock=None, shared_responses=None):
    if opts.exp_type == "preproc":
        gui = DefaultGUI(exp_info, opts, clock)
    elif opts.exp_type == "esam":
        gui = EsamGUI(exp_info, opts, shared_responses, clock)
    else:
        raise ValueError(f"Unknown exp_type: {opts.exp_type}")
    
    gui.run(sync)

class SharedClock:
      def __init__(self):
          self._start_time = time.perf_counter()
      
      def now(self):
          return time.perf_counter() - self._start_time
      
def euclidean_norm(nums):
     return math.sqrt(sum(x**2 for x in nums))