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
import os.path as osp
import multiprocessing
hv.extension('bokeh')

# %%capture
from tqdm.notebook import tqdm
tqdm().pandas()

from rtcap_lib.fMRI import load_fMRI_file, mask_fMRI_img, unmask_fMRI_img
from rtcap_lib.rt_functions import init_iGLM, rt_regress_vol, gen_polort_regressors
from rtcap_lib.rt_functions import init_EMA, rt_EMA_vol
from rtcap_lib.rt_functions import init_kalman, rt_kalman_vol
from rtcap_lib.rt_functions import rt_smooth_vol

from config import CAP_indexes, CAP_labels, CAPs_Path
from config import Mask_Path
from config import SVRs_Path, OUT_Dir
from config import TRAIN_OUT_Prefix as OUT_Prefix
from config import TRAIN_Path as Data_Path
from config import TRAIN_Motion_Path as Data_Motion_Path
n_CAPs      = len(CAP_indexes)
print(' + Data Path       : %s' % Data_Path)
print(' + Data Motion Path: %s' % Data_Motion_Path)
print(' + GM Ribbon Path  : %s' % Mask_Path)

DONT_USE_VOLS   = 10  # No Detrending should be done on non-steady state volumes
FIRST_VALID_VOL = 100 # It takes quite some time for detrending to become stable
POLORT = 2
FWHM = 4

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
# # Data Pre-processing

Data_Motion = np.loadtxt(Data_Motion_Path) #[np.arange(DONT_USE_VOLS,TRAIN_Nt),:]
nuisance_motion = Data_Motion - Data_Motion.mean(axis=0)
print('++ Motion Regressor Dimensions [%s]' % str(Data_Motion.shape))

nuisance_polort = gen_polort_regressors(POLORT,Data_Nt)

nuisance = np.concatenate((nuisance_polort,nuisance_motion),axis=1)
nuisance_labels = ['Polort'+str(i) for i in np.arange(POLORT)] + ['roll','pitch','yaw','dS','dL','dP']
n_regressors = nuisance.shape[1]
nuisance_DF = pd.DataFrame(nuisance,columns=nuisance_labels)
nuisance_DF.hvplot().opts(width=1000, legend_position='top', height=200)

Vols4Preproc = np.arange(DONT_USE_VOLS,Data_Nt)
print('++ Number of Volumes to pre-process [%d]: [min=%d, max=%d] (Python)' % (Vols4Preproc.shape[0], np.min(Vols4Preproc), np.max(Vols4Preproc)))

num_cores = 16

# +
n                  = 0
Data_FromAFNI      = Data_InMask #Nv,Nt
Data_EMA           = np.zeros(Data_FromAFNI.shape)
Data_iGLM          = np.zeros(Data_FromAFNI.shape)
Data_kalman        = np.zeros(Data_FromAFNI.shape)
Data_smooth        = np.zeros(Data_FromAFNI.shape)
do_EMA             = True
do_kalman          = True
do_smooth          = True
Regressor_Coeffs   = np.zeros((n_regressors,Data_Nv,Data_Nt))

pool    = multiprocessing.Pool(processes=num_cores)

for t in tqdm(np.arange(Data_Nt)):
    # 1) Discard Non-Steady State Volumes (so they don't influence estimates)
    # -----------------------------------------------------------------------
    if t not in Vols4Preproc:
        continue
    
    # 2) If this is first valid volume, do all initializations
    # --------------------------------------------------------
    if t == Vols4Preproc[0]:
        n, prev_iGLM = init_iGLM()
        S_x, S_P, \
        fPositDerivSpike, fNegatDerivSpike, kalmThreshold = init_kalman(Data_Nv,Data_Nt)
        #S = init_kalman(Data_Nv, Data_Nt)
        EMA_th, EMA_filt = init_EMA()
    else:
        n = n + 1
    
    # 3) EMA Filter
    # -------------
    ema_out, EMA_filt = rt_EMA_vol(n,t,EMA_th,Data_FromAFNI,EMA_filt, do_operation=do_EMA)
    Data_EMA[:,t] = np.squeeze(ema_out)
    # 4) iGLM
    # -------
    Data_iGLM[:,t],prev_iGLM, Bn = rt_regress_vol(n,Data_EMA[:,t][:,np.newaxis],nuisance[t,:][:,np.newaxis],prev_iGLM)
    Regressor_Coeffs[:,:,t]      = Bn.T
    
    # 5) Kalman Filter
    # ----------------
    Data_kalman[:,t],      \
    S_x[:,t], S_P[:,t],    \
    fPositDerivSpike[:,t], \
    fNegatDerivSpike[:,t] = rt_kalman_vol(n,t,Data_iGLM,
                    S_x[:,t-1],
                    S_P[:,t-1],
                    fPositDerivSpike[:,t-1],
                    fNegatDerivSpike[:,t-1], 
                    num_cores,DONT_USE_VOLS,pool,do_operation=do_kalman)
    
    # 6) Spatial Smoothing
    # --------------------
    Data_smooth[:,t] = rt_smooth_vol(Data_kalman[:,t],Mask_Img,fwhm=FWHM,do_operation=do_smooth)

pool.close()
# -

# ***
# # Data Standarization in Space

# Standarize CAPs in Space
# ========================
sc_CAPs = StandardScaler(with_mean=True, with_std=True)
CAPs_InMask = sc_CAPs.fit_transform(CAPs_InMask)

# Standarize Time series in Space
# ===============================
sc_Data_space  = StandardScaler(with_mean=True, with_std=True)
Data_norm      = sc_Data_space.fit_transform(Data_smooth)

# ***
# # Generation of Traning Labels (via Linear Regression + Z-scoring)

Vols4Training = np.arange(FIRST_VALID_VOL,Data_Nt)
print('++ Number of Volumes to SVR Training [%d]: [min=%d, max=%d] (Python)' % (Vols4Training.shape[0], np.min(Vols4Training), np.max(Vols4Training)))

# Initialize a Dictionary for results
Results = {}
for cap in CAP_labels:
    Results[cap]=[]
Results['R2'] = []

X_fmri = pd.DataFrame(CAPs_InMask, columns=CAP_labels)
for vol in tqdm(Vols4Training):
    Y_fmri = pd.Series(Data_norm[:,vol],name='V'+str(vol).zfill(4))
    lm     = linear_model.LinearRegression()
    model  = lm.fit(X_fmri,Y_fmri)
    for i,cap in enumerate(CAP_labels):
        Results[cap].append(lm.coef_[i])
    Results['R2'].append(lm.score(X_fmri,Y_fmri))
LinReg_Results       = pd.DataFrame(Results)
LinReg_Results['TR'] = Vols4Training

linear_regression_layout = (hv.Curve(LinReg_Results, kdims=['TR'],vdims=['R2'], label='R2 for Linear Regression on Training Data').opts(width=1500) + \
LinReg_Results.drop(['R2'],axis=1).hvplot(legend='top', label='Regression Coefficients for all CAPs', x='TR').opts(width=1500)).cols(1)
hv.save(linear_regression_layout,'./LinearRegression_Results.png',backend='matplotlib')
linear_regression_layout

# Becuase we don't want to make all CAPs equally plausible, I believe it makes sense to Z-score across all of them, and not on a CAP-by-CAP basis
# which happened to be my original approach based on LaConte's work (but his work only had one classifier)
# ================================================================================================================================================
[LR_Nt, LR_Ncaps]    = LinReg_Results[CAP_labels].shape
All_LinReg           = LinReg_Results[CAP_labels]
All_LinReg           = All_LinReg.values.reshape(LR_Ncaps*LR_Nt, order='F')
All_LinReg_Z         = zscore(All_LinReg)
All_LinReg_Z         = All_LinReg_Z.reshape((LR_Nt,LR_Ncaps), order='F')
LinReg_Zscores       = pd.DataFrame(All_LinReg_Z, columns=CAP_labels, index=LinReg_Results.index)
LinReg_Zscores['TR'] = Vols4Training
LinReg_Zscores.hvplot(legend='top', label='Z-scored training labels', x='TR').opts(width=1500)

# ***
# # Support Vector Regression (Training)

C       = 1.0
epsilon = 0.01
SVRs    = {}
for cap_lab in tqdm(CAP_labels):
    Training_Labels = LinReg_Zscores[cap_lab]
    mySVR = SVR(kernel='linear', C=C, epsilon=epsilon)
    mySVR.fit(Data_norm[:,Vols4Training].T,Training_Labels)
    SVRs[cap_lab] = mySVR

import pickle
pickle_out = open(SVRs_Path,"wb")
pickle.dump(SVRs, pickle_out)
pickle_out.close()

for variable, file_suffix in zip([Data_EMA, Data_iGLM, Data_kalman, Data_smooth, Data_norm], \
                          ['.pp01_EMA.nii','.pp02_iGLM.nii','.pp03_Kalman.nii','.pp04_Smooth.nii','.pp05_Norm.nii']):
    out = unmask_fMRI_img(variable, Mask_Img, osp.join(OUT_Dir,OUT_Prefix+file_suffix))

for i,lab in enumerate(nuisance_labels):
    data = Regressor_Coeffs[i,:,:]
    out = unmask_fMRI_img(data, Mask_Img, osp.join(OUT_Dir,OUT_Prefix+'.pp02_iGLM_'+lab+'.nii'))


