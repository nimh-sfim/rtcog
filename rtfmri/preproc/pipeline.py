import time
import sys
import os.path as osp
import multiprocessing as mp

import numpy as np
import pandas as pd

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '../../../')))

from preproc.preproc_steps import PreprocStep
from preproc.step_types import StepType
from utils.exceptions import VolumeOverflowError
from utils.core import initialize_kalman_pool
from utils.rt_functions import kalman_filter_mv
from utils.log import get_logger
from utils.fMRI import unmask_fMRI_img
from paths import OUTPUT_DIR

log = get_logger()

class Pipeline:
    """
    Real-time fMRI preprocessing pipeline.

    This class handles the initialization and execution of a customizable
    real-time fMRI preprocessing pipeline. Operates on a TR-by-TR basis.

    Parameters
    ----------
    options : Options
        Configuration object containing flags for which steps to run and what to save.
    Nt : int
        Number of TRs expected in the session.
    mask_Nv : int, optional
        Number of voxels in the brain mask.
    mask_img : nibabel.Nifti1Image, optional
        Binary brain mask used for reshaping and spatial operations.
    exp_type : str, optional
        Type of experiment being conducted (used for downstream logic).

    Attributes
    ----------
    steps : list
        List of preprocessing steps initialized from STEP_REGISTRY.
    Data_FromAFNI : np.ndarray
        Original incoming data from AFNI (Nv, Nt).
    Data_EMA : np.ndarray
        Data after EMA filtering.
    Data_iGLM : np.ndarray
        Data after iGLM nuisance regression.
    Data_kalman : np.ndarray
        Data after Kalman filtering.
    Data_smooth : np.ndarray
        Data after spatial smoothing.
    Data_norm : np.ndarray
        Data after spatial Z-scoring.
    Data_processed : np.ndarray
        Final processed data.
    motion_estimates : list
        Motion parameters across volumes.
    pool : multiprocessing.Pool
        Pool for parallel Kalman filtering (if enabled).
    nuisance_labels : list
        Labels for nuisance regressors (e.g., motion and polynomial terms).
    legendre_pols : np.ndarray
        Legendre polynomial regressors for detrending.
    """
    def __init__(self, options, Nt, mask_Nv=None, mask_img=None, exp_type=None):
        self.t = None
        self.n = None

        self.Nt = Nt
        self.mask_Nv = mask_Nv
        self.mask_img = mask_img
        self.exp_type = exp_type

        self._processed_tr = np.zeros((self.mask_Nv,1))

        self.motion_estimates = []
        
        self.save_orig = options.save_orig
        # self.save_all = options.save_all

        self.out_dir = options.out_dir
        self.out_prefix = options.out_prefix
        
        self.snapshot = options.snapshot

        self.Data_FromAFNI = None # np.array [Nv,Nt] for incoming data
        self.Data_processed = None

        self.step_registry = PreprocStep.registry
        self.step_opts = options.steps

        self.build_steps()
        self.run_funcs = [step.run for step in self.steps]

        self.FWHM = options.FWHM # FWHM for Spatial Smoothing in [mm]
        
        self.iGLM_motion = options.iGLM_motion
        self.iGLM_polort = options.iGLM_polort

        # If kalman needed, create a pool
        if StepType.KALMAN.value in self.steps:
            self.n_cores = options.n_cores
            self.pool = mp.Pool(processes=self.n_cores)
            if self.mask_Nv is not None:
                log.info(f'Initializing Kalman pool with {self.n_cores} processes ...')
                _ = self.pool.map(kalman_filter_mv, initialize_kalman_pool(self.mask_Nv, self.n_cores))
        else:
            self.n_cores = 0
            self.pool = None

    @property
    def processed_tr(self):
        return self._processed_tr
    
    @processed_tr.setter
    def processed_tr(self, value):
        if not isinstance(value, np.ndarray):
            log.error(f"pipeline.processed_tr must be a numpy array, but is of type {type(value)}")
            sys.exit(-1)
        if value.shape != (self.Nv, 1):
            log.error(f'pipeline.processed_tr has incorrect shape. Expected: {self.Nv, 1}. Actual: {value.shape}')
            sys.exit(-1)
        self._processed_tr = value
    
    def __del__(self):
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()
    
    def build_steps(self):
        """Build the pipeline"""
        self.steps = []
        for step in self.step_opts:
            name = step["name"].lower()
            if name not in self.step_registry:
                log.error(f'Unknown step: {name}')
                sys.exit(-1)
            StepClass = self.step_registry.get(name)
            if step["enabled"]:
                self.steps.append(StepClass(step["save"]))
        log.info(f"Steps used: {', '.join([step.name for step in self.steps])}")

    def process_first_volume(self, this_t_data):
        """Create empty structures"""
        self.Nv = len(this_t_data)
        log.info('Number of Voxels Nv=%d' % self.Nv)

        if StepType.SMOOTH.value in self.steps:
            if self.mask_Nv != self.Nv:
                log.error(f'Discrepancy across masks [data Nv = {self.Nv}, mask Nv = {self.mask_Nv}]')
                sys.exit(-1)

        self.Data_FromAFNI = np.zeros((self.Nv, self.Nt))
        self.Data_FromAFNI[:, self.t] = this_t_data
        log.debug(f'[t={self.t},n={self.n}] Init - Data_FromAFNI.shape {self.Data_FromAFNI.shape}')
    
        self.Data_processed = np.zeros((self.Nv, self.Nt)) # Final output
        
        for step in self.steps:
            step.start_step(self)

        return 1
    
    def process(self, t, n, motion, this_t_data):
        """Run full pipeline on a single TR"""  
        self.t = t
        self.n = n
        self.motion = motion

        if self.t == 0:
            return self.process_first_volume(this_t_data)
        if self.n == 0:
            self.Data_FromAFNI[:, self.t] = this_t_data
            return 1
        
        try:
            self.Data_FromAFNI[:, self.t] = this_t_data
        except IndexError:
            raise VolumeOverflowError()

        for func in self.run_funcs:
            self.processed_tr[:] = func(self)
        
        self.Data_processed[:, self.t] = self.processed_tr[:, 0]

        return self.processed_tr

    def final_steps(self):
        self.save_motion_estimates()

        if self.mask_img is None:
            log.warning(' final_steps = No additional outputs generated due to lack of mask.')
            return 1
        
        log.debug(' final_steps - About to write outputs to disk.')
        self.save_nifti_files()

        # If running snapshot test, save the variable states
        if self.snapshot:
            var_dict = {}
            for step in self.steps:
                var_dict.update(step.snapshot())
            var_dict.update({
                'Data_FromAFNI': self.Data_FromAFNI,
                'Data_processed': self.Data_processed
            })
        
            snap_path = osp.join(OUTPUT_DIR, f'new_snapshots.npz')
            np.savez(snap_path, **var_dict) 
            log.info(f'Snapshot saved to OUTPUT_DIR at {snap_path}')
        
    def save_motion_estimates(self):
        self.motion_estimates = [item for sublist in self.motion_estimates for item in sublist]
        log.info(f'motion_estimates length is {len(self.motion_estimates)}')
        self.motion_estimates = np.reshape(self.motion_estimates,newshape=(int(len(self.motion_estimates)/6),6))
        motion_path = osp.join(self.out_dir,self.out_prefix+'.Motion.1D')
        np.savetxt(
            motion_path, 
            self.motion_estimates,
            delimiter="\t"
        )
        log.info(f'Motion estimates saved to disk: [{motion_path}]')

    def save_nifti_files(self):
        out_vars   = [self.Data_processed]
        out_labels = ['.pp_Zscore.nii']

        if self.save_orig:
            out_vars.append(self.Data_FromAFNI)
            out_labels.append('.orig.nii')
        
        for variable, file_suffix in zip(out_vars, out_labels):
            unmask_fMRI_img(np.array(variable), self.mask_img, osp.join(self.out_dir,self.out_prefix+file_suffix))
        
        for step in self.steps:
            step.save_nifti(self)