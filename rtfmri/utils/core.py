import os.path as osp
from scipy.signal.windows import exponential
import numpy as np

def file_exists(path):
      if not osp.isfile(path):
         raise FileNotFoundError(f"File not found: {path}")
      return path


def create_win(M, center=0, tau=3):
    win = exponential(M, center, tau, False)
    print('++ Create Window: Window Values [%s]' % str(win))
    return win[:, np.newaxis]


def welford(k,x,M,S):
   # inputs np.array(Nv,)
   Mnext = M + (x-M)/k
   Snext = S + (x-M)*(x-Mnext)
   if k == 1:
      std = np.zeros(x.shape[0])
   else:
      std = np.sqrt(Snext/(k-1))
   return Mnext, Snext, std
