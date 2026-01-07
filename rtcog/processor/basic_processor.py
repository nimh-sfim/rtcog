import sys
import numpy as np

from rtcog.preproc.pipeline import Pipeline
from rtcog.utils.log import set_logger
from rtcog.utils.fMRI import load_fMRI_file

class BasicProcessor:
    """
    Class representing a real-time fMRI processor.

    This class handles the setup of the data processing stream, initialization of the
    preprocessing pipeline, and the management of incoming data from the scanner.

    Parameters
    ----------
    options : Options
        Configuration object containing experiment parameters (e.g., TR, number of volumes, paths).
    sync : SyncEvents
        Container for multiprocessing synchronization primitives used in experiment.
    """
    def __init__(self, options, sync, **kwargs):
        self.log = set_logger(options.debug, options.silent)

        self.sync = sync

        self.exp_type = options.exp_type

        self.n = 0 # Counter for number of volumes pre-processed (Start = 1)
        self.t = -1 # Counter for number of received volumes (Start = 0
        self.Nv= None # Number of voxels in data mask
        self.Nt = options.nvols # Number acquisitions
        self.TR = options.tr # TR [seconds]

        self.nvols_discard = options.discard # Number of volumes to discard from any analysis (won't enter pre-processing)

        self.this_motion = None

        if options.mask_path is None:
            self.sync.end.set()
            raise ValueError('No mask was provided!')
        else:
            try:
                self.mask_img  = load_fMRI_file(options.mask_path)
            except Exception as e:
                self.sync.end.set()
                raise RuntimeError(f'Error loading mask file: {e}')
            self.mask_Nv = int(np.sum(self.mask_img.get_fdata()))
            self.log.debug(f'Number of Voxels in user-provided mask: {self.mask_Nv}')

        self.pipe = Pipeline(options, self.Nt, self.mask_Nv, self.mask_img)        
        

    def _compute_TR_data_impl(self, motion, extra):
        """
        Process data for the current TR by passing it to the pipeline.

        Parameters
        ----------
        motion : list of list[float]
            List of 6-element motion parameter lists (one per TR).
        extra : list of list[float]
            List of voxel values for the current TR, where each sublist contains time series data for a voxel.

        Returns
        -------
        np.array
            pipeline.processed_tr (the processed data for this TR)
        """
        self.t += 1

        # Keep a record of motion estimates
        self.this_motion = [i[self.t] for i in motion]
        self.pipe.motion_estimates.append(self.this_motion)
        
        if len(self.this_motion) != 6:
            raise ValueError(f'Motion not read in correctly. Expected length: 6 | Actual length: {len(self.this_motion)}')
        
        this_t_data = np.array([e[self.t] for e in extra])
        del extra

        self.Nv = len(this_t_data)

        if self.t > 0:
            if len(this_t_data) != self.Nv:
                self.log.error(f'Extra data not read in correctly.')
                self.log.error(f'Expected length: {self.Nv} | Actual length: {len(this_t_data)}')
                raise ValueError('Extra data not read in correctly.')
        
        # Update n (only if not longer a discard volume)
        if self.t > self.nvols_discard - 1:
            self.n += 1
        return self.pipe.process(self.t, self.n, self.this_motion, this_t_data)

    def compute_TR_data(self, motion, extra):
        """
        Public wrapper for TR processing with logging.

        Parameters
        ----------
        motion : list of list[float]
            6 motion parameters per TR.
        extra : list of list[float]
            Voxel-wise time series.

        Returns
        -------
        int
            Always returns 1 (for compatibility with callback interface).
        """
        self._compute_TR_data_impl(motion, extra)
        self.log.info(f' - Time point [t={self.t}, n={self.n}]')
        return 1
    
    def end_run(self, save=True):
        """
        Finalize experiment and optionally save outputs.

        Parameters
        ----------
        save : bool
            Whether to save final outputs (default: True).
        """
        self.pipe.final_steps(save=save)
        self.sync.end.set()
