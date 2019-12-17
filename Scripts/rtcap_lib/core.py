from scipy.signal import exponential
import numpy as np

def create_win(M,center=0,tau=3):
    win = exponential(M,center,tau,False)
    print ('++ Create Window: Window Values [%s]' % str(win))
    return win

def unpack_extra(extra):
   n_elements = len(extra)
   if np.mod(n_elements,8) > 0:
      print('++ ERROR: Number of Elements in package is not a multiple of 8.')
      print(' +        Very likely "Vals to Send" is not All Data.')
      return None
   aux = np.array(extra)
   n_voxels = int(n_elements/8)
   aux = aux.reshape(n_voxels,8)
   #roi_id   = aux[:,0]
   #roi_i    = aux[:,1]
   #roi_j    = aux[:,2]
   #roi_k    = aux[:,3]
   #roi_x    = aux[:,4]
   #roi_y    = aux[:,5]
   #roi_z    = aux[:,6]
   roi_data = aux[:,7]
   return roi_data
   #return (roi_i,roi_j,roi_k,roi_data)