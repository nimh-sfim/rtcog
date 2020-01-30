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
# This notebook creates PNGs with surface views for the CAPs we are studying in this project. These images are used in the different dashboards

import os.path as osp
from nilearn.image import load_img, index_img
import sys
sys.path.append('/data/SFIMJGC/PRJ_rtCAPs/rtcaps/')
from rtcap_lib.utils.image import trim, paste_images, MNIvol_to_surf_pngs
from rtcap_lib.utils.exp_defs import CAP_indexes, CAP_labels, CAPS_DIR

# ***
# ### 2. Create Output Directory for Images and Video

VIDEO_DIR = osp.join(CAPS_DIR,'Video')
if osp.exists(VIDEO_DIR):
    print('++ INFO: Run Dir already exists [%s]' % VIDEO_DIR)
else:
    os.mkdir(VIDEO_DIR)
    print('++ INFO: Dir just created [%s]' % VIDEO_DIR)

# ***
# ### 3. Generate Individual Frames

Data_Path = osp.join(CAPS_DIR,'Frontier2013_CAPs.nii')
Data_IMGs = load_img(Data_Path)
Ncaps     = Data_IMGs.shape[3]
OUT_DIR   = osp.join(CAPS_DIR,'Video')
for i, (cap_idx,cap_label) in enumerate(zip(CAP_indexes,CAP_labels)):
    print('%d | %d | %s' % (i,cap_idx,cap_label))
    cap_img    = index_img(Data_IMGs,cap_idx)
    out_prefix = 'CAPs_'+cap_label
    MNIvol_to_surf_pngs(cap_img,out_prefix,OUT_DIR)
