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

# # Description: OVerall
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
from rtcap_lib.rt_functions import rt_snorm_vol


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
    
    if data_A_nt > data_B_nt:
        nt_diff = data_A_nt - data_B_nt
        data_A = data_A[:,nt_diff:]
        data_A_nv_c, data_A_nt_c = data_A.shape
        print('++ INFO: Dataset A (Corrected) Dimensions [Nv=%d,Nt=%d]' % (data_A_nv_c, data_A_nt_c))
    if data_B_nt > data_A_nt:
        nt_diff = data_B_nt - data_A_nt
        data_B = data_B[:,nt_diff:]
        data_B_nv_c, data_B_nt_c = data_B.shape
        print('++ INFO: Dataset B (Corrected) Dimensions [Nv=%d,Nt=%d]' % (data_B_nv_c, data_B_nt_c))
    # Compute spatial correlations
    r   = []
    for t in np.arange(data_A_nt):
        a = data_A[:,t]
        b = data_B[:,t]
        r.append(np.corrcoef(a,b)[0,1])
    return r


PRJDIR               = '/data/SFIMJGC/PRJ_rtCAPs/'
SBJ                  = 'PILOT03'
mask_path            = osp.join(PRJDIR,'PrcsData',SBJ,'D00_ScannerData','GMribbon_R4Feed.nii')
afni_preZscore_path  = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Training_Procpy',SBJ+'.results','errts.'+SBJ+'.tproject+orig.HEAD')
afni_path            = osp.join(PRJDIR,'PrcsData',SBJ,'D02_Training_Procpy',SBJ+'.results','errts.'+SBJ+'.tproject.Zscore.nii')
rtcaps_path          = osp.join(PRJDIR,'PrcsData',SBJ,'D00_ScannerData',SBJ+'_Training.pp_Zscore.nii')

# +
afni_preZscore_Img  = load_fMRI_file(afni_preZscore_path)
mask_Img            = load_fMRI_file(mask_path)

afni_preZscore_Data = mask_fMRI_img(afni_preZscore_Img, mask_Img)
Nv,Nt               = afni_preZscore_Data.shape
zData               = np.zeros(afni_preZscore_Data.shape)
for t in np.arange(Nt):
    aux  = afni_preZscore_Data[:,t]
    zaux = rt_snorm_vol(aux)
    zData[:,t] = np.squeeze(zaux)
# -

_ = unmask_fMRI_img(zData,mask_Img,afni_path)

Graph_Title   = 'Final: rtCAPs vs. AfniProc'
R = compute_spatial_corr(afni_path, rtcaps_path, mask_path)

(hv.Curve(R, label='Afni vs. rtCAPs').opts(width=1500, tools=['hover'])).opts(show_legend=True, title=Graph_Title)

import seaborn as sns

sns.distplot(R)


