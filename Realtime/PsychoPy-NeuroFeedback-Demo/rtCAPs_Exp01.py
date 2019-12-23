#!/usr/bin/env python

# This is a reimplementation of the default real-time feedback experiment
# distributed with AFNI, implemented in realtime_receiver.py, using WX and
# Matplotlib for generating the GUI and plotting the results.
#
# This replaces the default GUI toolkit with PsychoPy, and will draw the
# same results and shapes to a PsychoPy window, in a manner synchronous
# with the old toolkit.
#
# This will serve as a basis or a template to build neuro-feedback type
# of experiment that can get data from AFNI (through the 'afniInterfaceRT'
# module, also distributed here).

import sys
import logging
from   optparse import OptionParser

import numpy as np
import pandas as pd

import afniInterfaceRT as nf

from sklearn.preprocessing import StandardScaler
import pickle
import nibabel as nib
from nilearn.image import new_img_like


# Tests if Psychopy is available
# ==============================
try:
   from   psychopy import visual, core # , sound
   psychopyInstalled = 1
except ImportError:
   psychopyInstalled = 0


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

class DemoExperiment(object):

   def __init__(self, options):

      self.TR_data      = []
      self.acq = -1

      # Motion Related Containers
      self.DF_motion = pd.DataFrame(columns=['dS','dL','dP','roll','pitch','yaw'])
      self.DF_motion_path = "/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/RTtest/Online_Motion2.pkl"
      self.FD = []

      # Support Vector Regression Machines
      SVRs_Path = "/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/RTtest/Online_SVRs.pkl"
      SVRs_pickle_in = open(SVRs_Path,"rb")
      self.SVRs = pickle.load(SVRs_pickle_in)
      print('++ Support Vector Regression Objects loaded into memory')

      # Mask Stuff
      self.MASK_Path = "/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/RTtest/GMribbon_R4Feed.nii"
      self.MASK_Img  = nib.load(self.MASK_Path)
      [self.MASK_Nx, self.MASK_Ny, self.MASK_Nz] = self.MASK_Img.shape
      self.MASK_Nv = self.MASK_Img.get_data().sum()
      self.MASK_Vector = np.reshape(self.MASK_Img.get_data(),(self.MASK_Nx*self.MASK_Ny*self.MASK_Nz),          order='F')
      
      # CAPs Specific Information for this particular experiment
      self.CAP_Labels = ['VMed','VPol','VLat','DMN','SMot','Audi','ExCn','rFPa','lFPa']
      self.n_CAPs = len(self.CAP_Labels)

      # Save Incomming Data
      self.DF_data = None
      self.Nvoxels = None
      self.DF_data_path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/RTtest/Online_IncomingData2.pkl'

      # Results from Predictions
      self.DF_predictions = pd.DataFrame(columns=self.CAP_Labels)
      self.hits = []
      self.predictions_path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/RTtest/Online_Predictions2.pkl'

      self.dc_params    = []

      self.show_data    = options.show_data

      if options.fullscreen is None:
         self.fullscreen = False
      else:
         self.fullscreen = True
         print('++ Full Screen Requested')

      print ("++ Initializing experiment stimuli")
      self.setupExperiment()

   def setupExperiment(self):

      """create the GUI for display of the demo data:
         * Create the window
         * Create the different graphic elements
         * Potentially create the input objects
      """

      if self.fullscreen:
         self.exptWindow = visual.Window(fullscr=True, allowGUI=False)
      else:
         self.exptWindow = visual.Window([1280, 720], allowGUI=False)

      # For this demonstration experiement, set corners of the "active area" (where
      # we will "draw") to be a square in the middle of a 16:9 screen.
      self.nPlotPoints  = 10
      self.xMax         = 0.50625
      self.xMin         = self.xMax * -1.0
      self.xDelta       = (self.xMax - self.xMin) / (1.0 * self.nPlotPoints)
      self.yMax         = 0.9
      self.yMin         = self.yMax * -1.0
      self.yDelta       = (self.yMax - self.yMin) / (1.0 * self.nPlotPoints)

      # Now divide this area into a series of vertical rectangles that we will draw
      # to when we have results.
      self.stimAreaCorners    = [None] * self.nPlotPoints
      self.drawnCorners       = [None] * self.nPlotPoints

      for i in range(self.nPlotPoints):
         self.stimAreaCorners[i] = np.array ([[(self.xMin + (self.xDelta*(i+1))), self.yMin],
                                              [(self.xMin + (self.xDelta*(i+1))), self.yMin],
                                              [(self.xMin + (self.xDelta*(i+0))), self.yMin],
                                              [(self.xMin + (self.xDelta*(i+0))), self.yMin]])

         self.drawnCorners[i]    = self.stimAreaCorners[i]

         displayArea = visual.ShapeStim (self.exptWindow, vertices = self.stimAreaCorners[i],
                                         autoLog = False, fillColor = [1, 1, 1])

      self.exptWindow.flip()

   def runExperiment (self, data):

      """
         After data is received and processed by the 'compute_TR_data' routine,
         call this routine to update the display, or whatever stimulus is being
         generated for the experiment.  This update should be a consistent
         follow on to what was done in the 'setupExperiment' routine.
      """

      length = len(data)
      if length == 0:
         return

      if self.show_data:
         print('-- TR %d, demo value: %s' % (length, data[length - 1][0]))
      if True:
         if length > 10:
            bot = length - 10
         else:
            bot = 0
         pdata = [data[ind][0] for ind in range(bot, length)]

         # To update the rectangles to be drawn, with the results of the stimulus modeling, add
         # the new data to the base shapes (using the simple element-by-element addition done
         # by numpy's matrix opertions). Also, update the display area as every shape is updated
         # to avoid drawing artifacts, where vertices get incorrectly assigned to the area to be
         # drawn.
         for i in range(self.nPlotPoints):
            if (len(data) - 1 - i) > 0:
               plotIndex = self.nPlotPoints - 1 - i
               self.drawnCorners[plotIndex] = np.array ([
                                                 [0.0, (self.yDelta * data[len(data) - 1 - i][0])],
                                                 [0.0, 0.0],
                                                 [0.0, 0.0],
                                                 [0.0, (self.yDelta * data[len(data) - 1 - i][0])]
                                                       ]) + self.stimAreaCorners[plotIndex]

               displayArea = visual.ShapeStim (self.exptWindow, vertices = self.drawnCorners[plotIndex],
                                         autoLog = False, fillColor = [-1, -1, 1])
               displayArea.draw()

         self.exptWindow.flip()

   def final_steps(self):
      print('++ Entering Final Steps Function.')
      self.DF_predictions.to_pickle(self.predictions_path)
      print(' + Predictions saved to pickle file: %s' % self.predictions_path)

      # Save Motion Estimates
      self.DF_motion['FD'] = self.FD
      self.DF_motion.to_pickle(self.DF_motion_path)
      print(' + Motion saved to pickle file: %s' % self.DF_motion_path)

      # Save incoming data
      self.DF_data.to_pickle(self.DF_data_path)
      print(' + Incoming data saved to pickle file: %s' % self.DF_data_path)
      
   def compute_TR_data(self, motion, extra):

      """If writing to the serial port, this is the main function to compute
      results from motion and/or extras for the current TR and
      return it as an array of floats.
      Note that motion and extras are lists of time series of length nread,
      so processing a time series is easy, but a single TR requires extracting
      the data from the end of each list.
      The possible computations is based on data_choice, specified by the user
      option -data_choice.  If you want to send data that is not listed, just
      add a condition.
      ** Please add each data_choice to the -help.  Search for motion_norm to
      find all places to edit.
      return 2 items:
          error code:     0 on success, -1 on error
          data array:     (possibly empty) array of data to send
      """
      self.acq = self.acq + 1
      print("++ Entering compute TR data [%d]" % self.acq)
      # The order is as follows: [dS, dL, dP, roll, pitch, yaw]
      
      # 1) Extract the values in the ROI
      current_DATA = unpack_extra(extra)

      # 2) Initializations that can only be done after first volume is recevied
      if self.acq == 0:
         self.Nvoxels = len(current_DATA)
         self.DF_data = pd.DataFrame(columns=np.arange(self.Nvoxels))
      self.DF_data.loc[self.acq] = current_DATA

      # 2) Spatially normalize
      sc = StandardScaler(with_mean=True, with_std=True)
      current_DATA = sc.fit_transform(current_DATA[:,np.newaxis])
      print('CURRENT DATA Mean + std [%f +/- %f]' % (current_DATA.mean(),current_DATA.std()))
      
      # 3) Compute FD
      self.DF_motion.loc[self.acq] = motion
      if self.acq > 0:
         self.FD.append((self.DF_motion.tail(2).diff().tail(1).abs()*[1,1,1,50*np.pi/180,50*np.pi/180,50*np.pi/180]).sum(axis=1).values[0])
      else:
         self.FD.append(0)

      # 4) Compute Predictions
      aux_pred = []
      for cap_idx,cap_lab in enumerate(self.CAP_Labels):
         aux_pred.append(self.SVRs[cap_lab].predict(current_DATA.T)[0])
      self.DF_predictions.loc[self.acq] = aux_pred
      print('Predictions %s' % str(aux_pred))

      # 5) Compute Matches

      # Original code
      
      npairs = len(extra) // 2
      print(npairs)
      if npairs <= 0:
          print('** no pairs to compute diff_ratio from...')
          return None

      # modify extra array, setting the first half to diff_ratio
      for ind in range(npairs):
         a = extra[2 * ind]
         b = extra[2 * ind + 1]
         if a == 0 and b == 0:
            newval = 0.0
         else:
            newval = (a - b) / float(abs(a) + abs(b))

         # --------------------------------------------------------------
         # VERY data dependent: convert from diff_ratio to int in {0..10}
         # assume AFNI_data6 demo                             15 Jan
         # 2013

         # now scale [bot,inf) to {0..10}, where val>=top -> 10
         # AD6: min = -0.1717, mean = -0.1605, max = -0.1490

         bot = -0.17         # s620: bot = 0.008, scale = 43.5
         scale = 55.0        # =~ 1.0/(0.1717-0.149), rounded up
         if len(self.dc_params) == 2:
            bot = self.dc_params[0]
            scale = self.dc_params[1]

         val = newval - bot
         if val < 0.0:
            val = 0.0
         ival = int(10 * val * scale)
         if ival > 10:
            ival = 10

         extra[ind] = ival

         print('++ diff_ratio: ival = %d (from %s), (params = %s)' %
               (ival, newval, self.dc_params))

         # save data and process
         self.TR_data.append(extra[0:npairs])
         self.runExperiment(self.TR_data)

         return extra[0:npairs]    # return the partial list



def processExperimentOptions (self, options=None):

   """
       Process command line options for on-going experiment.
       Customize as needed for your own experiments.
   """

   usage = "%prog [options]"
   description = "AFNI real-time demo receiver with demo visualization."
   parser = OptionParser(usage=usage, description=description)

   parser.add_option("-d", "--debug", action="store_true",
            help="enable debugging output")
   parser.add_option("-v", "--verbose", action="store_true",
            help="enable verbose output")
   parser.add_option("-p", "--tcp_port", help="TCP port for incoming connections")
   parser.add_option("-S", "--show_data", action="store_true",
            help="display received data in terminal if this option is specified")
   parser.add_option("-w", "--swap", action="store_true",
            help="byte-swap numerical reads if set")
   parser.add_option("-f", "--fullscreen", action="store_true",
            help="run in fullscreen mode")

   return parser.parse_args(options)



def main():

   # 1) Read Input Parameters: port, fullscreen, etc..
   opts, args = processExperimentOptions(sys.argv)
   print('++ Options: %s' % str(opts))
   # 2) If Psychopy is not available, stop running
   if (psychopyInstalled == 0):
      print("")
      print("  *** This program requires the PsychoPy module.")
      print("  *** PsychoPy was not found in the PYTHONPATH.")
      print("  *** Please install PsychoPy before trying to use")
      print("      this module.")
      print("")
      return -1

   # 3) Set the logging level based on input parameters
   if opts.verbose and not opts.debug:
      nf.add_stderr_logger(level=logging.INFO)
   elif opts.debug:
      nf.add_stderr_logger(level=logging.DEBUG)
   
   # 4) Initialize the DemoExperiment Object
   print("++ Starting rtCAPs Experiment...")
   demo = DemoExperiment(opts)
   print(' + --> Completed.')
   
   # 5) create main interface
   print('++ Opening Communication Channel with AFNI....')
   receiver = nf.ReceiverInterface(port=opts.tcp_port, swap=opts.swap,
                                   show_data=opts.show_data)
   if not receiver:
      return 1

   # set signal handlers and look for data
   receiver.set_signal_handlers()  # require signal to exit

   # set receiver callback
   # At this point Receiver is still basically an empty container
   receiver.compute_TR_data  = demo.compute_TR_data
   receiver.final_steps      = demo.final_steps

   # prepare for incoming connections
   if receiver.RTI.open_incoming_socket():
      return 1

   rv = receiver.process_one_run()
   return rv



if __name__ == '__main__':
   sys.exit(main())

