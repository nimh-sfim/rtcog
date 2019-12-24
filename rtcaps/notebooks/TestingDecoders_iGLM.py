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
import sys, os
sys.path.insert(0, os.path.abspath('../'))
import config
import pickle
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
from rtcap_lib.svr_methods  import is_hit_method01
from rtcap_lib.core import create_win

from config import CAP_indexes, CAP_labels, CAPs_Path
from config import Mask_Path
from config import SVRs_Path, OUT_Dir
from config import TEST_OUT_Prefix as OUT_Prefix
from config import TEST_Path as Data_Path
from config import TEST_Motion_Path as Data_Motion_Path
n_CAPs      = len(CAP_indexes)
print(' + Data Path       : %s' % Data_Path)
print(' + Data Motion Path: %s' % Data_Motion_Path)
print(' + GM Ribbon Path  : %s' % Mask_Path)
print(' + SVR Models Path : %s' % SVRs_Path)

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
    Data_EMA[:,t], EMA_filt = rt_EMA_vol(n,t,EMA_th,Data_FromAFNI,EMA_filt, do_operation=do_EMA)
    
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
# # Support Vector Regression (Realtime Predictions)

SVRs_pickle_in = open(SVRs_Path,"rb")
SVRs = pickle.load(SVRs_pickle_in)
print('++ Support Vector Regression Objects loaded into memory')

Vols4Testing = np.arange(FIRST_VALID_VOL,Data_Nt)
Vols4Testing.shape

# +
hit_opts = {'method':               'method01',
            'vols2discard':         10,
            'vols2discard_postHit': 10,
            'z_th':                 2,
            'vols4hit':             2}

Realtime_Predictions      = {}
Vol_LastQEnded            = 0
WL = 4
DO_WINDOWS      = True
CONSIDER_MOTION = False
Hits_DF         = pd.DataFrame(False,columns=CAP_labels, index=np.arange(Data_Nt))
SVRscores_DF  = pd.DataFrame(columns=CAP_labels, index=np.arange(Data_Nt))
win_weights     = create_win(4)
for vol in tqdm(np.arange(Data_Nt)):
    # 1) Discard Initial Volumes
    # --------------------------
    if vol < hit_opts['vols2discard']:
        #print('++ Skipping Initial Volume [%d]' % vol)
        continue
        
    # 2) Create Voxel-wise Map (single-vol or win average)
    # ----------------------------------------------------
    if DO_WINDOWS == True:
        # Construct this acq input (via window averaging)
        vols_to_use   = (np.arange(vol-4,vol)+1)[::-1]                                 # For vol = 999 --> vols_to_use = 999 998 997 996
        weigthed_Data = Data_norm[:,vols_to_use] * win_weights
        current_DATA  = (weigthed_Data.sum(axis=1)/win_weights.sum())[:,np.newaxis].T
    else:
        current_DATA = (Data_norm[:,vol][:,np.newaxis]).T
    
    # 3) If motion is used in hit-decision, then extract and compute FD
    # ------------------------------------------------------------------
    if CONSIDER_MOTION == True:
        current_MOT  = Data_Mot.loc[vol]
        # Frame-wise Displacement
        if vol > 0:
            previous_MOT = Data_Mot.loc[vol-1]
            FD = np.abs(current_MOT['dS'] - previous_MOT['dS']) + \
                 np.abs(current_MOT['dL'] - previous_MOT['dL']) + \
                 np.abs(current_MOT['dP'] - previous_MOT['dP']) + \
                 ((50 * np.pi / 180) * np.abs(current_MOT['roll'] - previous_MOT['roll'])) + \
                 ((50 * np.pi / 180) * np.abs(current_MOT['pitch'] - previous_MOT['pitch'])) + \
                 ((50 * np.pi / 180) * np.abs(current_MOT['yaw'] - previous_MOT['yaw']))
        else:
            FD = None
    else:
        FD = 0
        current_MOT = 0
    
    # 4) Obtain predictions (SVRs)
    # ----------------------------
    aux_pred = []
    for cap_idx,cap_lab in enumerate(CAP_labels):
        aux_pred.append(SVRs[cap_lab].predict(current_DATA)[0])
    aux_pred = np.array(aux_pred)
    SVRscores_DF.loc[vol] = aux_pred
    
    # 5) Compute Hits
    if hit_opts['method'] == 'method01':
        matches, hit = is_hit_method01(vol, CAP_labels, hit_opts, SVRscores_DF, Realtime_Predictions,  Vol_LastQEnded)
    if hit != None:
        Vol_LastQEnded    = vol
        Hits_DF[hit][vol] = True
    # 6) Update records
    aux_dict = {'pred_per_cap':aux_pred,'matches':matches,'hit':hit, 'mot':current_MOT, 'FD':FD}
    Realtime_Predictions[vol] = aux_dict
Hits_DF.sum()
# -

SVRscores_DF.loc[np.arange(10,Data_Nt)].hvplot().opts(width=1500, toolbar='above', legend_position='top')

# ***
# # Save Average Hit Maps to Disk

for cap in CAP_labels:
    thisCAP_hits = Hits_DF[cap].sum()
    if thisCAP_hits > 0:
        thisCAP_Vols = []
        for vol in Hits_DF[Hits_DF[cap]==True].index:
            thisCAP_Vols.append(vol-np.arange(hit_opts['vols4hit']+1))
        thisCAP_Vols = [item for sublist in thisCAP_Vols for item in sublist]
        print('+ CAP [%s] was hit %d times. Contributing Vols: %s' % (cap,thisCAP_hits,str(thisCAP_Vols)))
        thisCAP_InMask  = Data_norm[:,thisCAP_Vols].mean(axis=1)
        unmask_fMRI_img(thisCAP_InMask, Mask_Img, osp.join(OUT_Dir,'RT_HitMap_'+cap+'.nii'))


