# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: psychopy_pyo_nilearn 0.6
#     language: python
#     name: psychopy_pyo_nilearn
# ---

# # Description: test smoothing step
#
# This notebook helped me evaluate the quality of the smoothing step. The algorithm programmed in the rtCAPs package was compared against that in afni 3dmerge under two different circumstances:
#
# * Data has not yet been de-meaned --> has small fluctuations over high values (the anatomical structure).
# * 3Data has been de-meaned --> small fluctuations around zero.
#
# Testing has been conducted on the Training run of subject PILOT03.

import os
import sys
import os.path as osp
import numpy as np
import holoviews as hv
hv.extension('bokeh')
sys.path.append('/data/SFIMJGC/PRJ_rtCAPs/rtcaps')
from rtcap_lib.fMRI import mask_fMRI_img, unmask_fMRI_img, load_fMRI_file


def compute_spatial_corr(data_A_path, data_B_path, mask_path, discard=100):
    data_A_img = load_fMRI_file(data_A_path)
    data_B_img = load_fMRI_file(data_B_path)
    mask_img   = load_fMRI_file(mask_path)
    
    # Data A
    data_A = mask_fMRI_img(data_A_img, mask_img)
    data_A = data_A[:,discard:]
    data_A_nv, data_A_nt = data_A.shape
    print('++ INFO: Dataset A Dimensions [Nv=%d,Nt=%d]' % (data_A_nv, data_A_nt))
    
    # Data B
    data_B = mask_fMRI_img(data_B_img, mask_img)
    data_B = data_B[:,discard:]
    data_B_nv, data_B_nt = data_B.shape
    print('++ INFO: Dataset A Dimensions [Nv=%d,Nt=%d]' % (data_B_nv, data_B_nt))
    
    # Compute spatial correlations
    r   = []
    for t in np.arange(data_A_nt):
        a = data_A[:,t]
        b = data_B[:,t]
        r.append(np.corrcoef(a,b)[0,1])
    return r


TestData_Dir       = '/data/SFIMJGC/PRJ_rtCAPs/PrcsData/PILOT03/D02_Training_Procpy/Smooth_Tests'
mask_path          = osp.join(TestData_Dir,'GMribbon_R4Feed.nii')

Graph_Title   = '4mm Smoothing'
afni_path     = osp.join(TestData_Dir,'afni_3dmerge_fwhm4.nii')
rtcaps_path   = osp.join(TestData_Dir,'rtcaps_all_fwhm4.nii')
R_withStructure = compute_spatial_corr(afni_path, rtcaps_path, mask_path)

afni_path     = osp.join(TestData_Dir,'afni_3dmerge_fwhm4.errts.nii')
rtcaps_path   = osp.join(TestData_Dir,'rtcaps_all_fwhm4.errts.nii')
R_withoutStructure = compute_spatial_corr(afni_path, rtcaps_path, mask_path)

(hv.Curve(R_withStructure, label='Data with Anatoical Structure: Afni vs. rtCaps').opts(width=1500, tools=['hover']) * \
hv.Curve(R_withoutStructure, label='Data without Anatoical Structure: Afni vs. rtCaps').opts(width=1500, tools=['hover'])).opts(show_legend=True, title=Graph_Title, ylim=(.99,1))


