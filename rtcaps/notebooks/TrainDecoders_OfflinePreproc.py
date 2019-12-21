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

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import holoviews as hv
import hvplot.pandas
from scipy.stats import zscore
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import config
hv.extension('bokeh')

# %%capture
from tqdm.notebook import tqdm
tqdm().pandas()

from rtcap_lib import load_fMRI_file, mask_fMRI_img

from config import CAP_indexes, CAP_labels, CAPs_Path
from config import TRAIN_Path as Data_Path
from config import TRAIN_Motion_Path as Data_Motion_Path
from config import Mask_Path
from config import SVRs_Path
n_CAPs      = len(CAP_indexes)

# Load Data in Memory
CAPs_Img     = load_fMRI_file(CAPs_Path)
Mask_Img     = load_fMRI_file(Mask_Path)
Data_Img    = load_fMRI_file(Data_Path)
Data_Motion = np.loadtxt(Data_Motion_Path)
print('++ CAPs Dimensions  : %s' % str(CAPs_Img.header.get_data_shape()))
print('++ DATA Dimensions  : %s' % str(Data_Img.header.get_data_shape()))
print('++ MASK Dimensions  : %s' % str(Mask_Img.header.get_data_shape()))
print('++ Motion Dimensions: %s' % str(Data_Motion.shape))

# Mask & Vectorize Data
CAPs_InMask = mask_fMRI_img(CAPs_Img,Mask_Img)
CAPs_InMask = CAPs_InMask[:, CAP_indexes]                            # Select only CAPs of intrest 
Data_InMask = mask_fMRI_img(Data_Img,Mask_Img)
[Data_Nv, Data_Nt] = Data_InMask.shape
print('++ Masked CAPs Dimensions  : %s' % str(CAPs_InMask.shape))
print('++ Masked Data Dimensions  : %s' % str(Data_InMask.shape))

# ***
# # Data Standarization in Space

# Standarize CAPs in Space
# ========================
sc_CAPs = StandardScaler(with_mean=True, with_std=True)
CAPs_InMask = sc_CAPs.fit_transform(CAPs_InMask)

# Standarize Time series in Space
# ===============================
sc_Data_space = StandardScaler(with_mean=True, with_std=True)
Data_InMask   = sc_Data_space.fit_transform(Data_InMask)

# ***
# # Generation of Traning Labels (via Linear Regression + Z-scoring)

# Initialize a Dictionary for results
Results = {}
for cap in CAP_labels:
    Results[cap]=[]
Results['R2'] = []

X_fmri = pd.DataFrame(CAPs_InMask, columns=CAP_labels)
for vol in tqdm(np.arange(Data_Nt)):
    Y_fmri = pd.Series(Data_InMask[:,vol],name='V'+str(vol).zfill(4))
    lm     = linear_model.LinearRegression()
    model  = lm.fit(X_fmri,Y_fmri)
    for i,cap in enumerate(CAP_labels):
        Results[cap].append(lm.coef_[i])
    Results['R2'].append(lm.score(X_fmri,Y_fmri))
LinReg_Results       = pd.DataFrame(Results)
LinReg_Results['TR'] = np.arange(Data_Nt)

linear_regression_layout = (hv.Curve(LinReg_Results, kdims=['TR'],vdims=['R2'], label='R2 for Linear Regression on Training Data').opts(width=1500) + \
LinReg_Results[CAP_labels].hvplot(legend='top', label='Regression Coefficients for all CAPs').opts(width=1500)).cols(1)
hv.save(linear_regression_layout,'./LinearRegression_Results.png',backend='matplotlib')
linear_regression_layout

# Becuase we don't want to make all CAPs equally plausible, I believe it makes sense to Z-score across all of them, and not on a CAP-by-CAP basis
# which happened to be my original approach based on LaConte's work (but his work only had one classifier)
# ================================================================================================================================================
All_LinReg = LinReg_Results[CAP_labels]
All_LinReg = All_LinReg.values.reshape(len(CAP_labels)*Data_Nt, order='F')
All_LinReg_Z = zscore(All_LinReg)
All_LinReg_Z = All_LinReg_Z.reshape((Data_Nt,len(CAP_labels)), order='F')
LinReg_Zscores = pd.DataFrame(All_LinReg_Z, columns=CAP_labels, index=LinReg_Results.index)
LinReg_Zscores['TR'] = LinReg_Results['TR']
LinReg_Zscores.drop(['TR'], axis=1).hvplot(legend='top', label='Z-scored training labels').opts(width=1500)

# ***
# # Support Vector Regression (Training)

C       = 1.0
epsilon = 0.01
SVRs    = {}
for cap_lab in tqdm(CAP_labels):
    Training_Labels = LinReg_Zscores[cap_lab]
    mySVR = SVR(kernel='linear', C=C, epsilon=epsilon)
    mySVR.fit(Data_InMask.T,Training_Labels)
    SVRs[cap_lab] = mySVR

import pickle
pickle_out = open(SVRs_Path,"wb")
pickle.dump(SVRs, pickle_out)
pickle_out.close()

# +
# load_fMRI_file?
# -


