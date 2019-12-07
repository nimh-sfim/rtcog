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
#     display_name: psychopy
#     language: python
#     name: psychopy
# ---

TRAIN_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/RTtest/rt.__002%reg3D_detrend+orig.HEAD'
MASK_Path         = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/RTtest/GMribbon_R4Feed.nii'
CAPs_Path         = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/RTtest/Frontier2013_CAPs_R4Feed.nii'
SVRs_Path  = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/D00_OriginalData/SVRs.pkl'
TRAIN_NVols_Discard = 90

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import holoviews as hv
import hvplot.pandas
from scipy.stats import zscore
from sklearn.svm import SVR
hv.extension('bokeh')

# %%capture
from tqdm.notebook import tqdm
tqdm().pandas()

# Load CAPs in same grid as functional (linearly interpolated)
# ============================================================
CAPs_Img  = nib.load(CAPs_Path)
[CAPs_Nx, CAPs_Ny, CAPs_Nz, n_CAPs] = CAPs_Img.shape
print ('+ CAPs Dimensions = %d CAPs | [%d,%d,%d]' % (n_CAPs, CAPs_Nx,CAPs_Ny,CAPs_Nz))

CAP_indexes = np.arange(n_CAPs)
CAP_labels  = np.array(['CAP'+str(cap).zfill(2) for cap in CAP_indexes])
rCAP_indexes = np.array([15,25,2,4,18,28,24,11,21])
rCAP_labels  = np.array(['VMed','VPol','VLat','DMN','SMot','Audi','ExCn','rFPa','lFPa'])
for i,ii in enumerate(rCAP_indexes):
    CAP_labels[ii] = rCAP_labels[i]

# Load FOV mask in same grid as functional (ensure it only cover areas that have data on input and CAPs)
# ======================================================================================================
MASK_Img  = nib.load(MASK_Path)
[MASK_Nx, MASK_Ny, MASK_Nz] = MASK_Img.shape
n_vox = MASK_Img.get_data().sum()
print ('+ Mask Dimensions = %d Voxels in mask | [%d,%d,%d]' % (n_vox, MASK_Nx,MASK_Ny,MASK_Nz))
print ('+ Mask Path: %s' % MASK_Path)

# Load Training Dataset
TRAIN_Img = nib.load(TRAIN_Path)
[TRAIN_Nx, TRAIN_Ny, TRAIN_Nz, TRAIN_Nt] = TRAIN_Img.shape
print ('+ Data Dimensions = [%d,%d,%d, Nvols = %d]' % (TRAIN_Nx, TRAIN_Ny, TRAIN_Nz, TRAIN_Nt))

# ***
# ## Vectorize all analysis inputs (Space is lost)

# +
# Vectorize All
# =============
MASK_Vector  = np.reshape(MASK_Img.get_data(),(MASK_Nx*MASK_Ny*MASK_Nz),          order='F')
CAPs_Vector  = np.reshape(CAPs_Img.get_data(),(CAPs_Nx*CAPs_Ny*CAPs_Nz, n_CAPs),  order='F')
TRAIN_Vector = np.reshape(TRAIN_Img.get_data(),(TRAIN_Nx*TRAIN_Ny*TRAIN_Nz, TRAIN_Nt), order='F')

# Minimize rounding errors
# ========================
CAPs_Vector  = CAPs_Vector.astype('float64')
TRAIN_Vector = TRAIN_Vector.astype('float64')

# Mask All (and keep only CAPs of interest)
# =========================================
CAPs_InMask = CAPs_Vector[MASK_Vector==1,:]
CAPs_InMask = CAPs_InMask[:,CAP_indexes]
TRAIN_InMask = TRAIN_Vector[MASK_Vector==1,:]
TRAIN_InMask = TRAIN_InMask[:,np.arange(TRAIN_NVols_Discard,TRAIN_Nt)] # Discard Volumes not yet stable (takes some time for on-line operations to stabilize)
[TRAIN_InMask_Nv, TRAIN_InMask_Nt] = TRAIN_InMask.shape
print('++ Final CAPs Template Dimensions: %d CAPs with %d voxels' % (CAPs_InMask.shape[1],CAPs_InMask.shape[0]))
print('++ Final Training Data Dimensions: %d Acqs with %d voxels' % (TRAIN_InMask_Nt, TRAIN_InMask_Nv))
# -

# ***
# # Data Standarization in Space

# Standarize CAPs in Space
# ========================
sc_CAPs = StandardScaler(with_mean=True, with_std=True)
CAPs_InMask = sc_CAPs.fit_transform(CAPs_InMask)

# Standarize Time series in Space
# ===============================
sc_TRAIN_space     = StandardScaler(with_mean=True, with_std=True)
TRAIN_InMask  = sc_TRAIN_space.fit_transform(TRAIN_InMask)

# ***
# # Generation of Traning Labels (via Linear Regression + Z-scoring)

# Initialize a Dictionary for results
Results = {}
for cap in CAP_labels:
    Results[cap]=[]
Results['R2'] = []

X_fmri = pd.DataFrame(CAPs_InMask, columns=CAP_labels)
for vol in tqdm(np.arange(TRAIN_InMask_Nt)):
    Y_fmri = pd.Series(TRAIN_InMask[:,vol],name='V'+str(vol).zfill(4))
    lm     = linear_model.LinearRegression()
    model  = lm.fit(X_fmri,Y_fmri)
    for i,cap in enumerate(CAP_labels):
        Results[cap].append(lm.coef_[i])
    Results['R2'].append(lm.score(X_fmri,Y_fmri))
LinReg_Results       = pd.DataFrame(Results)
LinReg_Results['TR'] = np.arange(TRAIN_InMask_Nt)

linear_regression_layout = (hv.Curve(LinReg_Results, kdims=['TR'],vdims=['R2'], label='R2 for Linear Regression on Training Data').opts(width=1500) + \
LinReg_Results.drop(['TR','R2'], axis=1).hvplot(legend='top', label='Regression Coefficients for all CAPs').opts(width=1500)).cols(1)
hv.save(linear_regression_layout,'./LinearRegression_Results.png',backend='matplotlib')
linear_regression_layout

# ***
# # Support Vector Regression (Training)

C       = 1.0
epsilon = 0.01
SVRs    = {}
for cap_lab in tqdm(rCAP_labels):
    Training_Labels = LinReg_Zscores[cap_lab]
    mySVR = SVR(kernel='linear', C=C, epsilon=epsilon)
    mySVR.fit(TRAIN_InMask.T,Training_Labels)
    SVRs[cap_lab] = mySVR

import pickle
pickle_out = open(SVRs_Path,"wb")
pickle.dump(SVRs, pickle_out)
pickle_out.close()
