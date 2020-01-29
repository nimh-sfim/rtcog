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

# # Description
#
# This notebook helped me evaluate the quality of the iGLM step. The algorithm programmed in the rtCAPs package was compared against three other scenarios:
#
# * 3dTproject using all 1000 datapoints
# * 3dTproject on a sequential basis (using in each instance all time points available until that point)
# * iGLM as programmed by Bob in the realtime_plugin
#
# Testing has been conducted on the Training run of subject PILOT03. All comparisons exists for removal of mean + linear + 6 motion parameters. For mean and linear, I did not do the comparisons agains the sequential 3dTproject case

import os
import sys
import os.path as osp
import numpy as np
import holoviews as hv
hv.extension('bokeh')
sys.path.append('/data/SFIMJGC/PRJ_rtCAPs/rtcaps')
from rtcap_lib.fMRI import mask_fMRI_img, unmask_fMRI_img, load_fMRI_file


def compute_Rs(afni_st_path, afni_iglm_path, rtcaps_iglm_path, mask_path):
    afni_st_img = load_fMRI_file(afni_st_path)
    afni_iGLM_img     = load_fMRI_file(afni_iglm_path)
    rtcaps_iGLM_img   = load_fMRI_file(rtcaps_iglm_path)
    mask_img          = load_fMRI_file(mask_path)
    
    # AFNI Static GLM
    afni_st_data = mask_fMRI_img(afni_st_img, mask_img)
    afni_st_data = afni_st_data[:,discard:]
    afni_st_nv, afni_st_nt = afni_st_data.shape
    print('++ INFO: Afni 3dTproject Dataset Dimensions [Nv=%d,Nt=%d]' % (afni_st_nv, afni_st_nt))
    
    # AFNI Incremental GLM
    afni_iGLM_data = mask_fMRI_img(afni_iGLM_img, mask_img)
    afni_iGLM_data = afni_iGLM_data[:,discard:]
    afni_iGLM_nv, afni_iGLM_nt = afni_iGLM_data.shape
    print('++ INFO: Afni iGLM Dataset Dimensions [Nv=%d,Nt=%d]' % (afni_iGLM_nv, afni_iGLM_nt))
    
    # rtcaps Incremental GLM
    rtcaps_iGLM_data = mask_fMRI_img(rtcaps_iGLM_img, mask_img)
    rtcaps_iGLM_data = rtcaps_iGLM_data[:,discard:]
    rtcaps_iGLM_nv, rtcaps_iGLM_nt = rtcaps_iGLM_data.shape
    print('++ INFO: rtCAPs iGLM Dimensions [Nv=%d,Nt=%d]' % (rtcaps_iGLM_nv, rtcaps_iGLM_nt))
    
    # Compute spatial correlations
    r_st2afni   = []
    r_st2rtcaps   = []
    r_rtcaps2afni = []
    for t in np.arange(afni_iGLM_nt):
        st = afni_st_data[:,t]
        af = afni_iGLM_data[:,t]
        jv = rtcaps_iGLM_data[:,t]
        r_st2afni.append(np.corrcoef(st,af)[0,1])
        r_st2rtcaps.append(np.corrcoef(st,jv)[0,1])
        r_rtcaps2afni.append(np.corrcoef(jv,af)[0,1])
    return r_st2afni, r_st2rtcaps, r_rtcaps2afni


TestData_Dir       = '/data/SFIMJGC/PRJ_rtCAPs/PrcsData/PILOT03/D02_Training_Procpy/GLM_Tests'
mask_path          = osp.join(TestData_Dir,'GMribbon_R4Feed.nii')
discard            = 100

# ### 1. When only removing the Mean

Graph_Title        = 'Mean Removal'
afni_st_path       = osp.join(TestData_Dir,'afni_3dTproject_Mean.nii')
afni_iGLM_path     = osp.join(TestData_Dir,'afni_iGLM_Mean.nii')
rtcaps_iGLM_path   = osp.join(TestData_Dir,'rtCAPs_iGLM_Mean.pp_iGLM.nii')

Mean_R_st2afni, Mean_R_st2rtcaps, Mean_R_rtcaps2afni = compute_Rs(afni_st_path,afni_iGLM_path,rtcaps_iGLM_path, mask_path)
(hv.Curve(Mean_R_st2afni, label='Static vs. Afni GLM').opts(width=1500, tools=['hover']) * \
hv.Curve(Mean_R_st2rtcaps,  label='Static vs. rtCAPs GLM').opts(width=1500, tools=['hover']) * \
hv.Curve(Mean_R_rtcaps2afni, label='rtCAPs GLM vs. Afni GLM').opts(width=1500, tools=['hover'])).opts(show_legend=True, title=Graph_Title, ylim=(.5,1))

# ***
# ### 2. When removing mean and linear trends

Graph_Title        = 'Mean + Linear Trend Removal'
afni_st_path       = osp.join(TestData_Dir,'afni_3dTproject_MeanLin.nii')
afni_iGLM_path     = osp.join(TestData_Dir,'afni_iGLM_MeanLin.nii')
rtcaps_iGLM_path   = osp.join(TestData_Dir,'rtCAPs_iGLM_MeanLin.pp_iGLM.nii')

Mean_R_st2afni, Mean_R_st2rtcaps, Mean_R_rtcaps2afni = compute_Rs(afni_st_path,afni_iGLM_path,rtcaps_iGLM_path, mask_path)
(hv.Curve(Mean_R_st2afni, label='Static vs. Afni GLM').opts(width=1500, tools=['hover']) * \
hv.Curve(Mean_R_st2rtcaps,  label='Static vs. rtCAPs GLM').opts(width=1500, tools=['hover']) * \
hv.Curve(Mean_R_rtcaps2afni, label='rtCAPs GLM vs. Afni GLM').opts(width=1500, tools=['hover'])).opts(show_legend=True, title=Graph_Title, ylim=(.5,1))

# ***
# ### 3. When removing mean + linear trends + motion

Graph_Title        = 'Mean + Linear Trend + 6 Motion Parameters Removal'
afni_st_path       = osp.join(TestData_Dir,'afni_3dTproject_MeanLinMot.nii')
afni_iGLM_path     = osp.join(TestData_Dir,'afni_iGLM_MeanLinMot.nii')
rtcaps_iGLM_path   = osp.join(TestData_Dir,'rtCAPs_iGLM_MeanLinMot.pp_iGLM.nii')

Mean_R_st2afni, Mean_R_st2rtcaps, Mean_R_rtcaps2afni = compute_Rs(afni_st_path,afni_iGLM_path,rtcaps_iGLM_path, mask_path)
(hv.Curve(Mean_R_st2afni, label='Static vs. Afni GLM').opts(width=1500, tools=['hover']) * \
hv.Curve(Mean_R_st2rtcaps,  label='Static vs. rtCAPs GLM').opts(width=1500, tools=['hover']) * \
hv.Curve(Mean_R_rtcaps2afni, label='rtCAPs GLM vs. Afni GLM').opts(width=1500, tools=['hover'])).opts(show_legend=True, title=Graph_Title, ylim=(.5,1), legend_position='bottom_right')

# ***
# ### 4. When removing mean + linear trends + motion | Static (in sequential way)

Graph_Title        = 'Mean + Linear Trend + 6 Motion Parameters Removal | vs. Sequential Static Gold Standard'
afni_st_path       = osp.join(TestData_Dir,'afni_3dTproject_MeanLinMot.sequential.nii')
afni_iGLM_path     = osp.join(TestData_Dir,'afni_iGLM_MeanLinMot.nii')
rtcaps_iGLM_path   = osp.join(TestData_Dir,'rtCAPs_iGLM_MeanLinMot.pp_iGLM.nii')

Mean_R_st2afni, Mean_R_st2rtcaps, Mean_R_rtcaps2afni = compute_Rs(afni_st_path,afni_iGLM_path,rtcaps_iGLM_path, mask_path)
(hv.Curve(Mean_R_st2afni, label='Static (sequential) vs. Afni GLM').opts(width=1500, tools=['hover']) * \
hv.Curve(Mean_R_st2rtcaps,  label='Static (sequential) vs. rtCAPs GLM').opts(width=1500, tools=['hover']) * \
hv.Curve(Mean_R_rtcaps2afni, label='rtCAPs GLM vs. Afni GLM').opts(width=1500, tools=['hover'])).opts(show_legend=True, title=Graph_Title, ylim=(.5,1))




