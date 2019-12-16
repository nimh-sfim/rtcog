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
hv.extension('bokeh')

# %%capture
from tqdm.notebook import tqdm
tqdm().pandas()

from rtcap_lib import load_fMRI_file, mask_fMRI_img, rt_regress_vol 
from rtcap_lib import gen_polort_regressors, unmask_fMRI_img, smooth_array
from rtcap_lib import kalman_filter_mv, apply_EMA_filter

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
FWHM = 6

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

import multiprocessing
print("Number of cpu : ", multiprocessing.cpu_count())
num_cores = 16

# +
Data_InMask_Step01 = np.zeros(Data_InMask.shape) # Data after EMA
Data_InMask_Step02 = np.zeros(Data_InMask.shape) # Data after iGLM
Data_InMask_Step03 = np.zeros(Data_InMask.shape) # Data after Kalman
Data_InMask_Step04 = np.zeros(Data_InMask.shape) # Data after Smoothing
Regressor_Coeffs   = np.zeros((n_regressors,Data_Nv,Data_Nt))

v_start = np.arange(0,Data_Nv,int(np.floor(Data_Nv/(num_cores-1)))).tolist()
v_end   = v_start[1:] + [Data_Nv]
pool    = multiprocessing.Pool(processes=num_cores)
for t in tqdm(np.arange(Data_Nt)):
    if t not in Vols4Preproc:
        continue
    if t == Vols4Preproc[0]:
        n                = 1 # Initialize counter
        prev_iGLM        = {} # Initialization for iGML
        S_x              = np.zeros((Data_Nv, Data_Nt))
        S_P              = np.zeros((Data_Nv, Data_Nt))
        fPositDerivSpike = np.zeros((Data_Nv, Data_Nt))
        fNegatDerivSpike = np.zeros((Data_Nv, Data_Nt))
        kalmThreshold    = np.zeros((Data_Nv, Data_Nt))
        # Initialization for EMA
        EMA_th      = 0.98
    else:
        n = n +1;
    
    # 1) EMA Filter
    if t == Vols4Preproc[0]:
        EMA_filt_in             = Data_InMask[:,t][:,np.newaxis]
        Data_InMask_Step01[:,t] = Data_InMask[:,t] - Data_InMask[:,t-1]
    else:
        [EMA_data_out, EMA_filt_out] = apply_EMA_filter(EMA_th,Data_InMask[:,t][:,np.newaxis],EMA_filt_in)
        EMA_filt_in                  = EMA_filt_out
        Data_InMask_Step01[:,t]      = np.squeeze(EMA_data_out)
    
    # 2) Remove Nuissance Regressors from data
    Yn = Data_InMask_Step01[:,t][:,np.newaxis]
    Fn = nuisance[t,:][:,np.newaxis]
    
    Yn_d,prev_iGLM, Bn      = rt_regress_vol(n,Yn,Fn,prev_iGLM)
    Data_InMask_Step02[:,t] = Yn_d
    Regressor_Coeffs[:,:,t] = Bn.T
    
    # 3) Low-Pass Filtering (Kalman Filter)
    if t > (Vols4Preproc[0] + 1):
        o_data, o_fPos, o_fNeg, o_S_x, o_S_P, o_voxels   = [],[],[],[],[],[]
        inputs = ({'d'   : Data_InMask_Step02[v_s:v_e,t],
                   'std' : np.std(Data_InMask_Step02[v_s:v_e,DONT_USE_VOLS:t+1], axis=1),
                   'S_x' : S_x[v_s:v_e,t-1],
                   'S_P' : S_P[v_s:v_e,t-1],
                   'S_Q' : 0.25 * np.power(np.std(Data_InMask_Step02[v_s:v_e,DONT_USE_VOLS:t+1], axis=1),2),
                   'S_R' : np.power(np.std(Data_InMask_Step02[v_s:v_e,DONT_USE_VOLS:t+1], axis=1),2),
                   'fPos': fPositDerivSpike[v_s:v_e,t-1],
                   'fNeg': fNegatDerivSpike[v_s:v_e,t-1],
                   'vox' : np.arange(v_s,v_e)}
                  for v_s,v_e in zip(v_start,v_end))
        res = pool.map(kalman_filter_mv,inputs)
        for j in np.arange(num_cores):
            o_data.append(res[j][0])
            o_fPos.append(res[j][1])
            o_fNeg.append(res[j][2])
            o_S_x.append(res[j][3])
            o_S_P.append(res[j][4])
            o_voxels.append(res[j][5])
        o_data   = [item for sublist in o_data   for item in sublist]
        o_fPos   = [item for sublist in o_fPos for item in sublist]
        o_fNeg   = [item for sublist in o_fNeg for item in sublist]
        o_S_x    = [item for sublist in o_S_x for item in sublist]
        o_S_P    = [item for sublist in o_S_P for item in sublist]
        o_voxels = [item for sublist in o_voxels for item in sublist]
        
        Data_InMask_Step03[:,t] = o_data
        S_x[:,t]                = o_S_x
        S_P[:,t]                = o_S_P
        fPositDerivSpike[:,t]   = o_fPos
        fNegatDerivSpike[:,t]   = o_fNeg
    
    Yn_df = Data_InMask_Step03[:,t]
    
    # 4) Spatial Smoothing
    Yn_d_vol    = unmask_fMRI_img(Yn_df, Mask_Img)
    Yn_d_vol_sm = smooth_array(Yn_d_vol,affine=Data_Img.affine, fwhm=FWHM)
    Yn_sm       = mask_fMRI_img(Yn_d_vol_sm, Mask_Img)
    
    # 5) Update Global Structures
    Data_InMask_Step04[:,t] = Yn_sm
pool.close()
pool.join()

# +
out = unmask_fMRI_img(Data_InMask_Step01, Mask_Img, osp.join(OUT_Dir,OUT_Prefix+'.pp_Step01.nii'))
out = unmask_fMRI_img(Data_InMask_Step02, Mask_Img, osp.join(OUT_Dir,OUT_Prefix+'.pp_Step02.nii'))
out = unmask_fMRI_img(Data_InMask_Step03, Mask_Img, osp.join(OUT_Dir,OUT_Prefix+'.pp_Step03.nii'))
out = unmask_fMRI_img(Data_InMask_Step04, Mask_Img, osp.join(OUT_Dir,OUT_Prefix+'.pp_Step04.nii'))

for i,lab in enumerate(nuisance_labels):
    data = Regressor_Coeffs[i,:,:]
    out = unmask_fMRI_img(data, Mask_Img, osp.join(OUT_Dir,OUT_Prefix+'.pp_Step02_'+lab+'.nii'))
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
Data_InMask_dn = sc_Data_space.fit_transform(Data_InMask_Step04)

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
    Y_fmri = pd.Series(Data_InMask_dn[:,vol],name='V'+str(vol).zfill(4))
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
    mySVR.fit(Data_InMask_dn[:,Vols4Training].T,Training_Labels)
    SVRs[cap_lab] = mySVR

import pickle
pickle_out = open(SVRs_Path,"wb")
pickle.dump(SVRs, pickle_out)
pickle_out.close()
