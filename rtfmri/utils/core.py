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


def initialize_kalman_pool(mask_Nv, n_cores):
      """Initialize pool up front to avoid delay later"""
      Nv = int(mask_Nv)
      return [
         {
               'd': np.zeros((Nv, 1)),
               'std': np.zeros((Nv)),
               'S_x': np.zeros((Nv)),
               'S_P': np.zeros((Nv)),
               'S_Q': np.zeros((Nv)),
               'S_R': np.zeros((Nv)),
               'fPos': np.zeros((Nv)),
               'fNeg': np.zeros((Nv)),
               'vox': np.zeros((Nv))
         }
         for _ in range(n_cores)
      ]
