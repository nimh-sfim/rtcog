import numpy as np
import pandas as pd
from multiprocessing.shared_memory import SharedMemory
import holoviews as hv
import panel as pn
import nibabel as nib
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
        self.Nt = config.Nt
        self.template_labels = config.template_labels
        self.Ntemplates = len(config.template_labels)

        self.t = config.matching_opts.match_start

        self.qa_state = None

        self.mask_img = config.mask_img
        self.m_x, self.m_y, self.m_z = self.mask_img.header.get_data_shape()
        self.mask_v = np.reshape(self.mask_img.get_fdata(),np.prod(self.mask_img.header.get_data_shape()), order='F')
        self.affine =  self.mask_img.affine
        
        self.brain_img = nib.Nifti1Image(np.zeros((self.mask_img.shape)), affine=self.affine)
        
        fig = plt.figure(figsize=(8, 6))
        plot_stat_map(self.brain_img, display_mode='ortho', draw_cross=False, figure=fig, bg_img=None)
        self.pane = pn.pane.Matplotlib(fig, dpi=150, tight=True, sizing_mode='scale_both')


    def update(self, t: int, data: np.ndarray, qa_state: QAState) -> None:
        self.t = t
        self.qa_state = qa_state
       
        if self.qa_state.qa_onsets and self.t == self.qa_state.qa_onsets[-1]:
            self.brain_img = self.arr_to_nifti(data)
            fig = plt.figure(figsize=(8, 6))
            plot_stat_map(self.brain_img, display_mode='ortho', draw_cross=False, figure=fig, bg_img=None)
            self.pane.object = fig

    def arr_to_nifti(self, data: np.ndarray) -> nib.nifti1.Nifti1Image:
        out = np.zeros((self.m_x * self.m_y * self.m_z,), dtype=np.float32)
        out[self.mask_v == 1] = data
        out = out.reshape((self.m_x, self.m_y, self.m_z), order='F')
        return type(self.mask_img)(out.astype(np.float32), affine=self.affine)



        
