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

TEST_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/TECH07_Run01_Testing.nii'
Motion_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/TECH07_Run01_Testing.Motion.1D'
MASK_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/rt.__002.mask.FB.nii'
CAPs_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/Frontier2013_CAPs_R4Feed.nii'
SVRs_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/Offline_SVRs.pkl'
DONT_USE_VOLS   = 10
FIRST_VALID_VOL = 50

import nibabel as nib
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import holoviews as hv
import hvplot.pandas
from scipy.stats import zscore
from sklearn.svm import SVR
from scipy.special import legendre
from numpy.linalg import cholesky, inv
hv.extension('bokeh')


# +
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 1e-10)

def iGLMVol(n,Yn,Fn,Dn,Cn,s2n):
    nv       = Yn.shape[0]
    nrBasFct = Fn.shape[0]  
    df = n - nrBasFct                        # Degrees of Freedom
    Dn = Dn + np.matmul(Yn,Fn.T)             # Eq. 17
    Cn = (((n-1)/n) * Cn) + ((1/n)*np.matmul(Fn,Fn.T))  # Eq. 18
    s2n = s2n + (Yn * Yn)            # Eq. 9 without the 1/n factor... see below
    if (is_pos_def(Cn) == True) and (n > nrBasFct + 2):
        Nn = cholesky(Cn).T
        iNn = inv(Nn.T)
        An    = (1/n) * np.matmul(Dn,iNn.T)  # Eq. 14
        Bn    = np.matmul(An,iNn)            # Eq. 16
    else:
        print ('%d non positive definite'% n)
        Bn = np.zeros((nv,nrBasFct))
    return Bn,Cn,Dn,s2n


# -

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

# Load TESTing Dataset
TEST_Img = nib.load(TEST_Path)
[TEST_Nx, TEST_Ny, TEST_Nz, TEST_Nt] = TEST_Img.shape
print ('+ Data Dimensions = [%d,%d,%d, Nvols = %d]' % (TEST_Nx, TEST_Ny, TEST_Nz, TEST_Nt))

# ***
# ## Vectorize all analysis inputs (Space is lost)

# +
# Vectorize All
# =============
MASK_Vector  = np.reshape(MASK_Img.get_data(),(MASK_Nx*MASK_Ny*MASK_Nz),          order='F')
CAPs_Vector  = np.reshape(CAPs_Img.get_data(),(CAPs_Nx*CAPs_Ny*CAPs_Nz, n_CAPs),  order='F')
TEST_Vector  = np.reshape(TEST_Img.get_data(),(TEST_Nx*TEST_Ny*TEST_Nz, TEST_Nt), order='F')

# Minimize rounding errors
# ========================
CAPs_Vector  = CAPs_Vector.astype('float64')
TEST_Vector  = TEST_Vector.astype('float64')

# Mask All (and keep only CAPs of interest)
# =========================================
CAPs_InMask = CAPs_Vector[MASK_Vector==1,:]
CAPs_InMask = CAPs_InMask[:,CAP_indexes]
TEST_InMask = TEST_Vector[MASK_Vector==1,:]
[TEST_InMask_Nv, TEST_InMask_Nt] = TEST_InMask.shape
print('++ Final CAPs Template Dimensions: %d CAPs with %d voxels' % (CAPs_InMask.shape[1],CAPs_InMask.shape[0]))
print('++ Final TESTing Data Dimensions: %d Acqs with %d voxels' % (TEST_InMask_Nt, TEST_InMask_Nv))

n_CAPs = CAPs_InMask.shape[1]
# -

# ***
# # Detrend Data (as if it happens online)

Motion = np.loadtxt(Motion_Path) #[np.arange(DONT_USE_VOLS,TEST_Nt),:]
nuisance_motion = Motion - Motion.mean(axis=0)
print('++ Motion Regressor Dimensions [%s]' % str(Motion.shape))

# Create Polort
polort          = 2
min             = -1.0
max             = 1.0
vols_to_discard = 10
#nuisance_polort = np.zeros((TEST_Nt-vols_to_discard, polort))
nuisance_polort = np.zeros((TEST_Nt, polort))
for n in range(polort):
    Pn = legendre(n)
    x = np.linspace(-1,1,TEST_Nt)
    nuisance_polort[:,n] = Pn(x).T
print('++ Polort Regressor Dimensions [%s]' % str(nuisance_polort.shape))

nuisance = np.concatenate((nuisance_polort,nuisance_motion),axis=1)
nuisance_DF = pd.DataFrame(nuisance,columns=['Polort'+str(i) for i in np.arange(polort)] + ['roll','pitch','yaw','dS','dL','dP'])
nuisance_DF.hvplot()

Vols4Detrending = np.arange(DONT_USE_VOLS,TEST_InMask_Nt)
Vols4Detrending.shape

TEST_InMask_d = np.zeros(TEST_InMask.shape)
for i,v in tqdm(enumerate(Vols4Detrending), total=Vols4Detrending.shape[0]):
    # This Acq Data + Nuisance
    n = i + 1
    Yn = TEST_InMask[:,v][:,np.newaxis]
    Fn = nuisance[v,:][:,np.newaxis]
    
    # Initializations for iGLM (only happens once)
    if v == Vols4Detrending[0]:
        L   = Fn.shape[0]
        Cn  = np.zeros((L,L), dtype='float64')
        Dn  = np.zeros((TEST_InMask_Nv, L), dtype='float64')
        s2n = np.zeros((TEST_InMask_Nv, 1), dtype='float64')
    
    # Detrend Volume
    Bn,Cn,Dn,s2n = iGLMVol(n,Yn,Fn,Dn,Cn,s2n)
    Yn_d = Yn - np.matmul(Bn,Fn)
    TEST_InMask_d[:,v] = np.squeeze(Yn_d)

output = np.zeros((MASK_Nx*MASK_Ny*MASK_Nz,TEST_InMask_d.shape[1]))
output[MASK_Vector==1,:] = TEST_InMask_d
output = np.reshape(output,(MASK_Nx,MASK_Ny,MASK_Nz,TEST_InMask_d.shape[1]),order='F')
output_img = nib.Nifti1Image(output,affine=MASK_Img.affine)
output_img.to_filename('/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/TECH07_Run01_TestingData.detrended.nii')

from nilearn.image import smooth_img

output_img_sm = smooth_img(output_img,6)

output_img_sm.to_filename('/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/TECH07_Run01_TestingData.detrended.smooth.nii')

SMOOTH_Vector = np.reshape(output_img_sm.get_data(),(TEST_Nx*TEST_Ny*TEST_Nz, TEST_Nt), order='F')
SMOOTH_Vector = SMOOTH_Vector.astype('float64')
SMOOTH_InMask = SMOOTH_Vector[MASK_Vector==1,:]
[SMOOTH_InMask_Nv, SMOOTH_InMask_Nt] = SMOOTH_InMask.shape
print('++ Final Training Data Dimensions: %d Acqs with %d voxels' % (SMOOTH_InMask_Nt, SMOOTH_InMask_Nv))

# ***
# # Data Standarization in Space

# Standarize CAPs in Space
# ========================
sc_CAPs = StandardScaler(with_mean=True, with_std=True)
CAPs_InMask = sc_CAPs.fit_transform(CAPs_InMask)

# Standarize Time series in Space
# ===============================
sc_TEST_space     = StandardScaler(with_mean=True, with_std=True)
#TEST_InMask_dn  = sc_TEST_space.fit_transform(TEST_InMask_d)
TEST_InMask_dn  = sc_TEST_space.fit_transform(SMOOTH_InMask)

# ***
# # Support Vector Regression (Realtime Predictions)

SVRs_pickle_in = open(SVRs_Path,"rb")
SVRs = pickle.load(SVRs_pickle_in)
print('++ Support Vector Regression Objects loaded into memory')

Vols4Testing = np.arange(FIRST_VALID_VOL,TEST_InMask_Nt)
Vols4Testing.shape

Initial_DoNothing_NVols   = 10
PostTrial_DoNothing_NVols = 0 
Realtime_Predictions      = {}
Z_Score_TH                = 1.75
NconsecutiveVols4Hit      = 2
Vol_LastQEnded            = 0
NHits                     = 0
Hits_DF        = pd.DataFrame(False,columns=rCAP_labels, index=np.arange(TEST_Nt))
Predictions_DF = pd.DataFrame(columns=rCAP_labels, index=np.arange(TEST_Nt))
for vol in tqdm(Vols4Testing):
    current_DATA = (TEST_InMask_dn[:,vol][:,np.newaxis]).T
    
    #current_MOT  = TEST_Mot.loc[vol]
    ## Frame-wise Displacement
    #if vol > 0:
    #    previous_MOT = TEST_Mot.loc[vol-1]
    #    FD = np.abs(current_MOT['dS'] - previous_MOT['dS']) + \
    #         np.abs(current_MOT['dL'] - previous_MOT['dL']) + \
    #         np.abs(current_MOT['dP'] - previous_MOT['dP']) + \
    #         ((50 * np.pi / 180) * np.abs(current_MOT['roll'] - previous_MOT['roll'])) + \
    #         ((50 * np.pi / 180) * np.abs(current_MOT['pitch'] - previous_MOT['pitch'])) + \
    #         ((50 * np.pi / 180) * np.abs(current_MOT['yaw'] - previous_MOT['yaw']))
    #else:
    #    FD = None
    
    FD = 0
    current_MOT = 0
    # Predictions
    aux_pred = []
    for cap_idx,cap_lab in enumerate(rCAP_labels):
        aux_pred.append(SVRs[cap_lab].predict(current_DATA)[0])
    aux_pred = np.array(aux_pred)
    Predictions_DF.loc[vol] = aux_pred
    # Matches
    num_matches = np.sum(aux_pred > Z_Score_TH)
    if num_matches > 0 :
        matches = [rCAP_labels[i] for i in np.where(aux_pred > Z_Score_TH)[0]]
    else:
        matches = None
    # Hit
    if (num_matches == 1) and (vol > Vol_LastQEnded + PostTrial_DoNothing_NVols + NconsecutiveVols4Hit):
        #print('Volume with potential hit: %d,%s' % (vol,matches))
        this_hit = matches[0]
        present = np.repeat(False,NconsecutiveVols4Hit-1)
        for ii,i in enumerate(np.arange(vol-NconsecutiveVols4Hit+1, vol)):
            relevant_matches = Realtime_Predictions[i]['matches']
            #print('  %d,%s' % (i,relevant_matches))
            if (relevant_matches is not None) and (this_hit in relevant_matches):
                present[ii] = True
        if np.all(present):
            hit = this_hit
            Vol_LastQEnded = vol
            Hits_DF[hit][vol] = True
            NHits = NHits + 1
            #print('++ HIT %d [vol=%d,CAP=%s]' % (NHits,vol+NTEST_Vols,hit))
        else:
            hit = None
    else:
        hit = None
    #if vol > Initial_DoNothing_NVols & vol > NconsecutiveVols4Hit
    # Save
    aux_dict = {'pred_per_cap':aux_pred,'matches':matches,'hit':hit, 'mot':current_MOT, 'FD':FD}
    Realtime_Predictions[vol] = aux_dict
#Hits_DF.index=np.arange(TEST_Nt)+NTEST_Vols
Hits_DF.sum()

Predictions_DF.loc[Vols4Testing].hvplot().opts(width=1500)

Hits_DF[Hits_DF['DMN']==True]


