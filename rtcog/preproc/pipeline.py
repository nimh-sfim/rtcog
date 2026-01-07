import sys
import os.path as osp
import numpy as np
from typing import Optional
from nibabel.nifti1 import Nifti1Image

from rtcog.preproc.preproc_steps import PreprocStep
from rtcog.preproc.step_types import StepType
from rtcog.utils.exceptions import VolumeOverflowError
from rtcog.utils.options import Options
from rtcog.utils.log import get_logger
from rtcog.utils.fMRI import unmask_fMRI_img
from rtcog.paths import OUTPUT_DIR

log = get_logger()

class Pipeline:
    """
    Realtime fMRI preprocessing pipeline.

    This class implements a pipeline design pattern for modular preprocessing of fMRI data.
    Data flows through a series of configurable steps (e.g., smoothing, normalization),
    allowing flexible analysis workflows. Each step is a separate object that can be
    enabled/disabled and configured independently.

    Attributes
    ----------
    Nt : int
        Total number of expected volumes (TRs) in the session.
    mask_Nv : int
        Number of voxels in the brain mask.
    mask_img : nibabel.Nifti1Image or None
        Brain mask image used.
    Nv : int
        Number of voxels in the data. Equal to mask_Nv.
    steps : list of PreprocStep
        Initialized preprocessing step objects.
    run_funcs : list of callable
        Preprocessing functions to apply to each TR during processing.
    motion_estimates : list
        Flattened list of motion parameters across volumes.
    Data_FromAFNI : np.ndarray or None
        Raw input data from AFNI with shape (Nv, Nt).
    Data_processed : np.ndarray or None
        Fully processed data with shape (Nv, Nt).
    processed_tr : np.ndarray
        Last processed TR as a column vector (Nv, 1).
    out_dir : str
        Directory where outputs will be written.
    out_prefix : str
        Filename prefix for output files.
    save_orig : bool
        Whether to save the original, unprocessed incoming volumes.
    snapshot : bool
        Whether to save a debug snapshot of internal variables.
    """
    def __init__(self, options: Options, Nt: int, mask_Nv: int, mask_img: Nifti1Image = None):
        """
        Initialize the Pipeline object and prepare for real-time fMRI processing.

        Parameters
        ----------
        options : Options
            Parsed configuration object with pipeline flags and settings.
        Nt : int
            Total number of TRs expected during the session.
        mask_Nv : int
            Number of voxels included in the brain mask.
        mask_img : nibabel.Nifti1Image, optional
            Brain mask used. 
        """
        self.t = None
        self.n = None

        self.Nt = Nt
        self.mask_Nv = mask_Nv
        self.mask_img = mask_img
        self.Nv = 0

        self._processed_tr = np.zeros((self.mask_Nv,1))

        self.motion_estimates = []
        
        self.save_orig = options.save_orig

        self.out_dir = options.out_dir
        self.out_prefix = options.out_prefix
        
        self.snapshot = options.snapshot

        self.Data_FromAFNI = None # np.array (Nv,Nt) for incoming data
        self.Data_processed = None

        self.step_registry = PreprocStep.registry
        self.step_opts = options.steps

        self.build_steps()
        self.run_funcs = [step.run for step in self.steps]

    @property
    def processed_tr(self) -> np.ndarray:
        """np.ndarray: Last processed TR as a column vector of shape (Nv, 1)."""
        return self._processed_tr
    
    @processed_tr.setter
    def processed_tr(self, value: np.ndarray) -> None:
        """
        Setter for processed_tr, ensuring correct shape and type.

        Parameters
        ----------
        value : np.ndarray
            Column vector of shape (Nv, 1) representing the processed TR.

        Raises
        ------
        SystemExit
            If the input is not a NumPy array or has the wrong shape.
        """
        if not isinstance(value, np.ndarray):
            raise ValueError(f"pipeline.processed_tr must be a numpy array, but is of type {type(value)}")
        if value.shape != (self.mask_Nv, 1):
            log.error(f'pipeline.processed_tr has incorrect shape. Expected: {self.mask_Nv, 1}. Actual: {value.shape}')
            raise ValueError(f'pipeline.processed_tr has incorrect shape. Expected: {self.mask_Nv, 1}. Actual: {value.shape}')
        self._processed_tr = value
    
    def build_steps(self) -> None:
        """
        Construct the list of preprocessing steps based on user options.

        This method uses the registered step types in `PreprocStep.registry`
        and only instantiates steps that are marked as "enabled" in the config.
        """
        self.steps = []
        for step in self.step_opts:
            name = step["name"].lower()
            if name not in self.step_registry:
                log.error(f'Unknown step: {name}')
                sys.exit(-1)
            StepClass = self.step_registry.get(name)
            if not step.get("enabled", False):
                continue
            save = step.pop("save", False)
            step.pop("enabled", None)
            step.pop("name", None)

            kwargs = step.copy()
            if name == StepType.KALMAN.value:
                kwargs["mask_Nv"] = self.mask_Nv
            
            self.steps.append(StepClass(save=save, **kwargs))
        log.info(f"Steps used: {', '.join([step.name for step in self.steps])}")

    def process_first_volume(self, this_t_data: np.ndarray) -> None:
        """
        Initialize processing pipeline on the first volume.

        Parameters
        ----------
        this_t_data : np.ndarray
            A 1D array of data for the first TR, with length equal to the number of voxels.

        Raises
        ------
        SystemExit
            If the number of voxels in `this_t_data` does not match the expected mask size.
        """
        self.Nv = len(this_t_data)
        if self.mask_Nv != self.Nv:
            raise ValueError(f'Discrepancy across masks [data Nv = {self.Nv}, mask Nv = {self.mask_Nv}]')

        self.Data_FromAFNI = np.zeros((self.Nv, self.Nt))
        self.Data_processed = np.zeros((self.Nv, self.Nt))
        self._processed_tr = np.zeros((self.Nv, 1))

        for step in self.steps:
            step.start_step(self)

    def process(self, t: int, n: int, motion: list[float], this_t_data: np.ndarray) -> Optional[np.ndarray]:
        
        """
        Run full processing pipeline on a single TR.

        Parameters
        ----------
        t : int
            Time index for current volume.
        n : int
            Index used to determine whether this volume is discarded (0) or kept (non-zero).
        motion : list or array-like
            Motion parameters for the current volume.
        this_t_data : np.ndarray
            A 1D array of voxel data for the current time point.

        Returns
        -------
        np.ndarray or None
            The processed data for the current time point as a column vector,
            or `None` if the volume is discarded.

        Raises
        ------
        VolumeOverflowError
            If the time index exceeds the expected number of time points.
        """
        self.t = t
        self.n = n
        self.motion = motion

        if t == 0:
            self.process_first_volume(this_t_data)

        # Store raw data
        try:
            self.Data_FromAFNI[:, self.t] = this_t_data
        except IndexError:
            raise VolumeOverflowError()

        # Skip processing during discard volumes
        if self.n == 0:
            return None

        # Start with raw data
        self.processed_tr = this_t_data[:, np.newaxis]

        # Apply each preprocessing step in sequence
        # Each step modifies the data in-place for efficiency
        for func in self.run_funcs:
            self.processed_tr[:] = func(self)

        # Save output
        self.Data_processed[:, self.t] = self.processed_tr[:, 0]

        return self.processed_tr


    def final_steps(self, save=True) -> None:
        """
        Run finalization steps after all volumes are processed.

        This includes saving motion estimates, writing processed NIfTI files, 
        and optionally saving a snapshot of internal variables for testing/debugging.
        """
        for step in self.steps:
            step.end_step(self)
        
        if not save:
            return
            
        self.save_motion_estimates()

        if self.mask_img is None:
            # TODO: decide if I should require mask_img or not, since I do upstream in Options, so would have to adjust that
            log.warning(' final_steps = No additional outputs generated due to lack of mask.')
            return
        
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
        
    def save_motion_estimates(self) -> None:
        """
        Flatten and save motion estimates to disk.

        Output file is saved to: `self.out_dir/self.out_prefix + '.Motion.1D'`
        """
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

    def save_nifti_files(self) -> None:
        """
        Save processed and (optionally) original fMRI data to NIfTI format.

        Files are saved using `unmask_fMRI_img`, reconstructing full brain volumes
        from masked data and writing to: `self.out_dir/self.out_prefix + <suffix>`
        """
        out_vars   = [self.Data_processed]
        out_labels = ['.pp_Final.nii']

        if self.save_orig:
            out_vars.append(self.Data_FromAFNI)
            out_labels.append('.orig.nii')
        
        for variable, file_suffix in zip(out_vars, out_labels):
            unmask_fMRI_img(np.array(variable), self.mask_img, osp.join(self.out_dir,self.out_prefix+file_suffix))
        
        for step in self.steps:
            step.save_nifti(self)