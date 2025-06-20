import time
import os.path as osp
import math

def file_exists(path):
      if not osp.isfile(path):
         raise FileNotFoundError(f"File not found: {path}")
      return path

def euclidean_norm(nums):
     return math.sqrt(sum(x**2 for x in nums))


class SharedClock:
      def __init__(self):
          self._start_time = time.perf_counter()
      
      def now(self):
          return time.perf_counter() - self._start_time