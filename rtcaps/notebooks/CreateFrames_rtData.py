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
# This notebook takes the Hit Maps output by rtcaps_match.py when running on "esam" mode and generates PNGs of their surface view in MNI space
#
# For this the notebook assumes the following files do exists:
#
# * REF2MNI Transformation matrix
# * The location of the hit maps
#

import os.path as osp
import glob
import ntpath
import os
import numpy as np
from nilearn.image import load_img
import sys
sys.path.append('/data/SFIMJGC/PRJ_rtCAPs/rtcaps/')
from rtcap_lib.utils.exp_defs import PRJDIR
from rtcap_lib.utils.image    import MNIvol_to_surf_pngs

SBJ = 'PILOT05'
RUN = 'Run03'

# ***
# ### 1. Bring Frames of interest to MNI space (needed for display in surface)

SBJ_DIR    = osp.join(PRJDIR,'PrcsData',SBJ)
WORK_DIR   = osp.join(SBJ_DIR,'D01_OriginalData')
ORIG_DIR   = osp.join(SBJ_DIR,'D00_ScannerData')
OUT_DIR    = osp.join(PRJDIR,'Dashboards','SurfViews')
HitFiles   = glob.glob(WORK_DIR + '/'+SBJ+'_'+RUN+'.Hit*[0123456789].nii')
HitFiles.sort()
HitFiles

MNI_Files = {}
for file_path in HitFiles:
    filename = ntpath.basename(file_path)
    dirname  = ntpath.dirname(file_path)
    [prefix,middle,extension] = filename.split('.')
    
    zpad_path    = osp.join(dirname,prefix+'.'+middle+'.zpad.'+extension)
    zpad_MNIpath = osp.join(dirname,prefix+'.'+middle+'.zpad.MNI.'+extension)
    abox_MNIpath = osp.join(dirname,prefix+'.'+middle+'.zpad.MNI.abox.'+extension)
    REF2MNI_path = osp.join(ORIG_DIR,'REF2MNI.Xaff12.1D')
    # Zero Padding
    cmd_zeropad = '3dZeropad -overwrite -R 20 -L 20 -A 20 -P 20 -I 20 -S 20 -prefix '+ zpad_path + ' ' + file_path
    # Bring to MNI Space
    cmd_org2mni = '3dAllineate -1Dmatrix_apply ' + REF2MNI_path + ' -input ' + zpad_path + ' -prefix ' + zpad_MNIpath
    # Remove zero padded
    cmd_rmorig  = 'rm ' + zpad_path
    # Refit 
    cmd_refit   = '3drefit -space MNI ' + zpad_MNIpath
    # Autobox
    cmd_abox    = '3dAutobox -prefix ' + abox_MNIpath + ' ' + zpad_MNIpath
    # Remove MNI no autobox
    cmd_rmmni   = 'rm ' + zpad_MNIpath
    
    for command in [cmd_zeropad, cmd_org2mni,cmd_rmorig, cmd_refit]:
        print(command)
        os.system(command)
    MNI_Files[(prefix,middle)] = zpad_MNIpath

# ***
# ## 2. Generate PNGs with all surface views

for (prefix,middle) in MNI_Files.keys():
    file_path  = MNI_Files[(prefix,middle)]
    print('+ Working on %s' % str(file_path))
    image      = load_img(file_path)
    out_prefix = prefix + '.' + middle
    MNIvol_to_surf_pngs(image,out_prefix,OUT_DIR, threshold=0, vmax=4)


