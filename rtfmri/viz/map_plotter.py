import numpy as np
import panel as pn
from nibabel.nifti1 import Nifti1Image
from nilearn.plotting import plot_epi
from nilearn.plotting import plot_stat_map
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rtfmri.viz.streaming_config import StreamerConfig, QAState
from rtfmri.utils.fMRI import unmask_fMRI_img

class MapPlotter:
    """Plot brain maps at moment of each hit"""
    data_key = "tr_data"

    def __init__(self, config: StreamerConfig):
        self._Nt = config.Nt
        self._template_labels = config.template_labels
        self._Ntemplates = len(config.template_labels)

        self._t = config.matching_opts.match_start

        self._qa_state = None

        self._mask_img = config.mask_img
        self._m_x, self._m_y, self._m_z = self._mask_img.header.get_data_shape()
        self._mask_v = np.reshape(self._mask_img.get_fdata(),np.prod(self._mask_img.header.get_data_shape()), order='F')
        self._affine =  self._mask_img.affine
        
        self._brain_img = Nifti1Image(np.zeros((self._mask_img.shape)), affine=self._affine)
        
        self._fig = plt.figure(figsize=(7, 6))
        plot_stat_map(self._brain_img, display_mode='ortho', draw_cross=False, figure=self._fig, bg_img=None)
        self.pane = pn.pane.Matplotlib(self._fig, dpi=150, tight=True, sizing_mode='scale_both')


    def update(self, t: int, data: np.ndarray, qa_state: QAState) -> None:
        self._t = t
        self._qa_state = qa_state
       
        if self._qa_state.qa_onsets and self._t == self._qa_state.qa_onsets[-1]:
            self._brain_img = self._arr_to_nifti(data)
            # fig = plt.figure(figsize=(8, 6))
            self._fig.clear()
            plot_stat_map(self._brain_img, display_mode='ortho', draw_cross=False, figure=self._fig, bg_img=None)
            self.pane.object = self._fig

    def _arr_to_nifti(self, data: np.ndarray) -> Nifti1Image:
        out = np.zeros((self._m_x * self._m_y * self._m_z,), dtype=np.float32)
        out[self._mask_v == 1] = data
        out = out.reshape((self._m_x, self._m_y, self._m_z), order='F')
        return Nifti1Image(out.astype(np.float32), affine=self._affine)
    
    def close(self):
        plt.close(self._fig)



        
