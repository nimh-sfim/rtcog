import warnings
import numpy as np
import panel as pn
from nibabel.nifti1 import Nifti1Image
from nilearn.plotting import plot_stat_map
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rtcog.viz.streaming_config import StreamingConfig
from rtcog.viz.plotter import Plotter
from rtcog.utils.sync import QAState

class MapPlotter(Plotter):
    """
    Plot brain activation maps when each 'hit' occurs.

    This class generates brain plots using data masked by a mask image.
    It updates the plot only when a new QA onset is detected.
    """
    data_key = "tr_data"

    def __init__(self, config: StreamingConfig):
        """
        Initialize a MapPlotter object for visualizing brain activation maps.

        This sets up internal state including the brain mask and a placeholder
        (empty) brain image. A matplotlib figure is created and embedded in a
        Panel pane for real-time visualization.

        Parameters
        ----------
        config : StreamingConfig
            Configuration object containing streaming-related information.
        """
        super().__init__(config)
        self._mask_img = config.mask_img
        self._m_x, self._m_y, self._m_z = self._mask_img.header.get_data_shape()
        self._mask_v = np.reshape(self._mask_img.get_fdata() > 0, -1, order='F')
        self._affine =  self._mask_img.affine
        
        self._brain_img = Nifti1Image(np.zeros((self._mask_img.shape)), affine=self._affine)
        self._last_map_t = None
        
        self._fig = plt.figure(figsize=(5, 4))

        with warnings.catch_warnings(): # Ignore nilearn warnings
            warnings.simplefilter("ignore", UserWarning)
            plot_stat_map(self._brain_img, display_mode='ortho', draw_cross=False, figure=self._fig, bg_img=None)

        self.pane = pn.pane.Matplotlib(self._fig, dpi=150, tight=True, sizing_mode='scale_both')
    
    def should_update(self, t: int, qa_state: QAState) -> bool:
        """
        Whether or not the plot should be updated.
        
        Parameters
        ----------
        t : int
            Current TR.
        qa_state : QAState
            QA-related metadata for the current TR.
        Returns
        ----------
        bool
            Whether the plot should be updated.
        """
        return t in qa_state.qa_onsets

    def update(self, t: int, data: np.ndarray, qa_state: QAState) -> None:
        """
        Update the brain plot if a new QA onset (hit) is detected.

        Parameters
        ----------
        t : int
            Current TR.
        data : np.ndarray
            1D array containing brain activation values within the mask.
        qa_state : QAState
            Object holding information about QA onsets and offsets.
        """
        if qa_state.qa_onsets:
            latest_onset = qa_state.qa_onsets[-1]
            if self._last_map_t != latest_onset:
                self._last_map_t = latest_onset
                self._brain_img = self._arr_to_nifti(data)
                self._fig.clear()
                plot_stat_map(
                    self._brain_img,
                    display_mode='ortho',
                    draw_cross=False,
                    figure=self._fig,
                    bg_img=None
                )
                self.pane.object = self._fig

    def _arr_to_nifti(self, data: np.ndarray) -> Nifti1Image:
        """
        Convert a 1D masked brain data array to a NIfTI image.

        Parameters
        ----------
        data : np.ndarray
            1D array containing brain activation values within the mask.

        Returns
        -------
        Nifti1Image
            A 3D NIfTI image.
        """
        out = np.zeros((self._m_x * self._m_y * self._m_z,), dtype=np.float32)
        out[self._mask_v == 1] = data
        out = out.reshape((self._m_x, self._m_y, self._m_z), order='F')
        return Nifti1Image(out.astype(np.float32), affine=self._affine)
    
    def close(self):
        """
        Close the matplotlib figure to release resources.
        """
        plt.close(self._fig)



        
