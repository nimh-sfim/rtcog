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
import pickle
import os.path as osp
hv.extension('bokeh')

# %%capture
from tqdm.notebook import tqdm
tqdm().pandas()

from rtcap_lib import load_fMRI_file, mask_fMRI_img, rt_regress_vol 
from rtcap_lib import gen_polort_regressors, unmask_fMRI_img, smooth_array
from rtcap_lib import kalman_filter_mv, apply_EMA_filter
from rtcap_lib import create_win, is_hit_method01

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
        weigthed_Data = Data_InMask_dn[:,vols_to_use] * win_weights
        current_DATA  = (weigthed_Data.sum(axis=1)/win_weights.sum())[:,np.newaxis].T
    else:
        current_DATA = (Data_InMask_dn[:,vol][:,np.newaxis]).T
    
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
        thisCAP_InMask  = Data_InMask_dn[:,thisCAP_Vols].mean(axis=1)
        unmask_fMRI_img(thisCAP_InMask, Mask_Img, osp.join(OUT_Dir,'RT_HitMap_'+cap+'.nii'))


