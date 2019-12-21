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

import pandas as pd
import numpy as np
import nibabel as nib
import os.path as osp
from nilearn.masking import apply_mask, unmask
from nilearn.plotting import plot_glass_brain
import xarray as xr
import matplotlib.pyplot as plt
import hvplot.pandas
import holoviews as hv
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn import linear_model
from scipy.stats import zscore
from sklearn.svm import SVR
hv.extension('bokeh')

# %%capture
from tqdm.notebook import tqdm
tqdm().pandas()

# Manually select a few CAPs of interest
# ========================================
CAP_indexes = np.arange(30)
CAP_labels  = np.array(['CAP'+str(cap).zfill(2) for cap in CAP_indexes])
rCAP_indexes = np.array([15,25,2,4,18,28,24,11,21])
rCAP_labels  = np.array(['VMed','VPol','VLat','DMN','SMot','Audi','ExCn','rFPa','lFPa'])
for i,ii in enumerate(rCAP_indexes):
    CAP_labels[ii] = rCAP_labels[i]

# ***
# ## Load Data (as it after pre-processing)

# Data Selection
PRJDIR='/data/SFIMJGC_HCP7T/PRJ_rtCAPs/'
SUBDIR='D00_OriginalData'
#RUN='SBJ04_Rest'
SBJ='TECH06'

# Load CAPs in same grid as functional (linearly interpolated)
# ============================================================
CAPs_Path = osp.join(PRJDIR,'PrcsData',SBJ,SUBDIR,'Frontier2013_CAPs_R4Feed.nii')
CAPs_Img  = nib.load(CAPs_Path)
[CAPs_Nx, CAPs_Ny, CAPs_Nz, n_CAPs] = CAPs_Img.shape
print ('+ CAPs Dimensions = %d CAPs | [%d,%d,%d]' % (n_CAPs, CAPs_Nx,CAPs_Ny,CAPs_Nz))

# Load FOV mask in same grid as functional (ensure it only cover areas that have data on input and CAPs)
# ======================================================================================================
MASK_Path = osp.join(PRJDIR,'PrcsData',SBJ,SUBDIR,'GMribbon_R4Feed.nii')
MASK_Img  = nib.load(MASK_Path)
[MASK_Nx, MASK_Ny, MASK_Nz] = MASK_Img.shape
n_vox = MASK_Img.get_data().sum()
print ('+ Mask Dimensions = %d Voxels in mask | [%d,%d,%d]' % (n_vox, MASK_Nx,MASK_Ny,MASK_Nz))
print ('+ Mask Path: %s' % MASK_Path)

TRAIN_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/D00_OriginalData/TrainingSet.nii'
TEST_Path  = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/D00_OriginalData/TestingSet.nii'

TRAIN_Img = nib.load(TRAIN_Path)
[TRAIN_Nx, TRAIN_Ny, TRAIN_Nz, TRAIN_Nt] = TRAIN_Img.shape
print ('+ Data Dimensions = [%d,%d,%d, Nvols = %d]' % (TRAIN_Nx, TRAIN_Ny, TRAIN_Nz, TRAIN_Nt))

TEST_Img = nib.load(TEST_Path)
[TEST_Nx, TEST_Ny, TEST_Nz, TEST_Nt] = TEST_Img.shape
print ('+ Data Dimensions = [%d,%d,%d, Nvols = %d]' % (TEST_Nx, TEST_Ny, TEST_Nz, TEST_Nt))

# +
# Load motion parameters
# ======================
# -

# ***
# ## Vectorize all analysis inputs (Space is lost)

# +
# Vectorize All
# =============
MASK_Vector  = np.reshape(MASK_Img.get_data(),(MASK_Nx*MASK_Ny*MASK_Nz),          order='F')
CAPs_Vector  = np.reshape(CAPs_Img.get_data(),(CAPs_Nx*CAPs_Ny*CAPs_Nz, n_CAPs),  order='F')
TRAIN_Vector = np.reshape(TRAIN_Img.get_data(),(TRAIN_Nx*TRAIN_Ny*TRAIN_Nz, TRAIN_Nt), order='F')
TEST_Vector  = np.reshape(TEST_Img.get_data(),(TEST_Nx*TEST_Ny*TEST_Nz, TEST_Nt), order='F')

# Minimize rounding errors
# ========================
CAPs_Vector  = CAPs_Vector.astype('float64')
TRAIN_Vector = TRAIN_Vector.astype('float64')
TEST_Vector  = TEST_Vector.astype('float64')

# Mask All (and keep only CAPs of interest)
# =========================================
CAPs_InMask = CAPs_Vector[MASK_Vector==1,:]
CAPs_InMask = CAPs_InMask[:,CAP_indexes]
TRAIN_InMask = TRAIN_Vector[MASK_Vector==1,:]
TEST_InMask = TEST_Vector[MASK_Vector==1,:]
print('++ Final CAPs Template Dimensions: %d CAPs with %d voxels' % (CAPs_InMask.shape[1],CAPs_InMask.shape[0]))
print('++ Final Training Data Dimensions: %d Acqs with %d voxels' % (TRAIN_InMask.shape[1], TRAIN_InMask.shape[0]))
print('++ Final Training Data Dimensions: %d Acqs with %d voxels' % (TEST_InMask.shape[1], TEST_InMask.shape[0]))

n_CAPs = CAPs_InMask.shape[1]
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
sc_TEST_space     = StandardScaler(with_mean=True, with_std=True)
TEST_InMask  = sc_TEST_space.fit_transform(TEST_InMask)

# ***
# # Generation of Traning Labels (via Linear Regression + Z-scoring)

# Initialize a Dictionary for results
Results = {}
for cap in CAP_labels:
    Results[cap]=[]
Results['R2'] = []

X_fmri = pd.DataFrame(CAPs_InMask, columns=CAP_labels)
for vol in tqdm(np.arange(TRAIN_Nt)):
    Y_fmri = pd.Series(TRAIN_InMask[:,vol],name='V'+str(vol).zfill(4))
    lm     = linear_model.LinearRegression()
    model  = lm.fit(X_fmri,Y_fmri)
    for i,cap in enumerate(CAP_labels):
        Results[cap].append(lm.coef_[i])
    Results['R2'].append(lm.score(X_fmri,Y_fmri))
LinReg_Results       = pd.DataFrame(Results)
LinReg_Results['TR'] = np.arange(TRAIN_Nt)

(hv.Curve(LinReg_Results, kdims=['TR'],vdims=['R2'], label='R2 for Linear Regression on Training Data').opts(width=1500, tools=['hover'], padding=0.01) + \
LinReg_Results.drop(['TR','R2'], axis=1).hvplot(legend='top', label='Regression Coefficients for all CAPs').opts(width=1500)).cols(1)

# Steve LaConte uses the Z-score version of the regression coefficients as the labels
# ===================================================================================
LinReg_Zscores = pd.DataFrame(columns=LinReg_Results.columns, index=LinReg_Results.index)
for col in LinReg_Results.columns:
    LinReg_Zscores[col] = zscore(LinReg_Results[col])
LinReg_Zscores['TR'] = LinReg_Results['TR']
LinReg_Zscores.drop(['TR','R2'], axis=1).hvplot(legend='top', label='Z-scored training labels').opts(width=1500)

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
pickle_out = open("/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/D00_OriginalData/SVRs.pkl","wb")
pickle.dump(SVRs, pickle_out)
pickle_out.close()

# ***
# # Support Vector Regression (Realtime Predictions)

Initial_DoNothing_NVols   = 10
PostTrial_DoNothing_NVols = 0 
Realtime_Predictions      = {}
Z_Score_TH                = 1.75
NconsecutiveVols4Hit      = 2
Vol_LastQEnded            = 0
NHits                     = 0
Hits_DF        = pd.DataFrame(False,columns=rCAP_labels, index=np.arange(TEST_Nt))
Predictions_DF = pd.DataFrame(columns=rCAP_labels, index=np.arange(TEST_Nt))
for vol in tqdm(np.arange(TEST_Nt)):
    current_DATA = (TEST_InMask[:,vol][:,np.newaxis]).T
    
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
            #print('++ HIT %d [vol=%d,CAP=%s]' % (NHits,vol+NTRAIN_Vols,hit))
        else:
            hit = None
    else:
        hit = None
    #if vol > Initial_DoNothing_NVols & vol > NconsecutiveVols4Hit
    # Save
    aux_dict = {'pred_per_cap':aux_pred,'matches':matches,'hit':hit, 'mot':current_MOT, 'FD':FD}
    Realtime_Predictions[vol] = aux_dict
#Hits_DF.index=np.arange(TEST_Nt)+NTRAIN_Vols
Hits_DF.sum()

Predictions_DF.hvplot().opts(width=1500)

# # Create GlassBrain Views of Matches

A = np.array([[i-2,i-1,i] for i in Hits_DF[Hits_DF['Audi']==True].index])
A_Img = unmask(TEST_InMask[:,A.flatten()].mean(axis=1), MASK_Img)

plot_glass_brain(A_Img)



Hits_DF[Hits_DF['lFPa']==True]

A = np.array([[i-2,i-1,i] for i in Hits_DF[Hits_DF['lFPa']==True].index])
A.flatten()

kwargs = {'cumulative': True}
sns.distplot(Predictions_DF.values.flatten())

Hits_DF.sum()

Questioning_DF = Predictions_DF.copy()
Questioning_DF[Predictions_DF<Z_Score_TH]=0
Questioning_DF.hvplot(legend='top', kind='scatter').opts(width=1500)

Z_Score_TH = 2
Questioning_DF = Predictions_DF.copy()
Questioning_DF.drop(np.arange(0,1000), inplace=True)
Questioning_DF.reset_index(inplace=True)
Questioning_DF.drop('index', axis=1, inplace=True)
[Q_nvols, Q_ncaps] = Questioning_DF.shape
HITS = np.repeat(None, Q_nvols)
for vol in np.arange(2,Q_nvols):
    ThisVol_CAP = None
    ThisVol_originalPredictions = Questioning_DF.loc[vol]
    ThisVol_originalPredictions[ThisVol_originalPredictions < Z_Score_TH] = 0
    ThisVol_Hits = np.sum(ThisVol_originalPredictions > 0)    
    if ThisVol_Hits == 1:
        HITS[vol]  = ThisVol_originalPredictions[ThisVol_originalPredictions > 0].index[0]
    if (HITS[vol] is not None) and (HITS[vol] == HITS[vol-1]) and (HITS[vol]==HITS[vol-2]):
        print('HIT [%d] --> %s' % (vol,HITS[vol]))

aux_pred < Z


