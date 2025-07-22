import time
import os.path as osp
import math
from multiprocessing import Event, Value
from ctypes import c_int

from rtfmri.utils.sync import SyncEvents

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

class SharedClock:
      def __init__(self):
          self._start_time = time.perf_counter()
      
      def now(self):
          return time.perf_counter() - self._start_time
      
def euclidean_norm(nums):
     return math.sqrt(sum(x**2 for x in nums))