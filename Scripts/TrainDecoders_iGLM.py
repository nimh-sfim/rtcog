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

from rtcap_lib import load_fMRI_file, mask_fMRI_img, rt_regress_vol, gen_polort_regressors, unmask_fMRI_img, smooth_array, kalman_filter

from config import CAP_indexes, CAP_labels, CAPs_Path
from config import Mask_Path
from config import SVRs_Path
n_CAPs      = len(CAP_indexes)

# +
Data_Path        = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/TECH07_Run01_Training.nii'
Data_Motion_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/TECH07_Run01_Training.Motion.1D'

DONT_USE_VOLS   = 10  # No Detrending should be done on non-steady state volumes
FIRST_VALID_VOL = 100 # It takes quite some time for detrending to become stable
POLORT = 2
FWHM = 6
# -

# Load Data in Memory
CAPs_Img     = load_fMRI_file(CAPs_Path)
Mask_Img     = load_fMRI_file(Mask_Path)
Data_Img    = load_fMRI_file(Data_Path)
Data_Motion = np.loadtxt(Data_Motion_Path)
print('++ CAPs Dimensions  : %s' % str(CAPs_Img.header.get_data_shape()))
print('++ DATA Dimensions  : %s' % str(Data_Img.header.get_data_shape()))
print('++ MASK Dimensions  : %s' % str(Mask_Img.header.get_data_shape()))
print('++ Motion Dimensions: %s' % str(Data_Motion.shape))

type(Mask_Img)

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

# + active=""
# # Smoothing before Kalman
# Data_InMask_Preprocessed = np.zeros(Data_InMask.shape)
# Regressor_Coeffs         = np.zeros((n_regressors,Data_Nv,Data_Nt))
# Data_InMask_Filtered = np.zeros(Data_InMask_Preprocessed.shape)
# for t in tqdm(np.arange(Data_Nt)):
#     # Initializations
#     if t not in Vols4Preproc:
#         continue
#     if t == Vols4Preproc[0]:
#         n                = 1 # Initialize counter
#         prev_iGLM        = {} # Initialization for iGML
#         S                = np.repeat({'Q':0, 'P':0, 'x':0, 'R':0},Data_Nv) # Initializations for Kalman Filter
#         fPositDerivSpike = np.zeros(Data_Nv)
#         fNegatDerivSpike = np.zeros(Data_Nv)
#         kalmThreshold    = np.zeros(Data_Nv)
#     else:
#         n = n +1;
#
#     Yn = Data_InMask[:,t][:,np.newaxis]
#     Fn = nuisance[t,:][:,np.newaxis]
#     
#     # Remove Nuissance Regressors from data
#     Yn_d,prev_iGLM, Bn = rt_regress_vol(n,Yn,Fn,prev_iGLM)
#     
#     # Smooth Image
#     Yn_d_vol    = unmask_fMRI_img(Yn_d, Mask_Img)
#     Yn_d_vol_sm = smooth_array(Yn_d_vol,affine=Data_Img.affine, fwhm=FWHM)
#     Yn_sm       = mask_fMRI_img(Yn_d_vol_sm, Mask_Img)
#     
#     # Update Global Structures
#     Data_InMask_Preprocessed[:,t] = Yn_sm   
#     Regressor_Coeffs[:,:,t] = Bn.T
#     
#     # Low Pass Filter
#     if t > (Vols4Preproc[0] + 1):
#         for voxel in np.arange(Data_Nv):
#             this_voxel_preprocessed_ts = Data_InMask_Preprocessed[voxel,np.arange(DONT_USE_VOLS,t)]
#             this_voxel_preprocessed_ts_STD = np.std(this_voxel_preprocessed_ts) 
#             # Case 1 (DCM) R/Q = 4 --> Cut-Off ~ 0.1Hz
#             S[voxel]['Q'] = 0.25*np.power(this_voxel_preprocessed_ts_STD,2)
#             S[voxel]['R'] = np.power(this_voxel_preprocessed_ts_STD,2)
#             # Case 2 (PSC, SVM)
#             #S[voxel]['Q'] = np.power(this_voxel_preprocessed_ts_STD,2)
#             #S[voxel]['R'] = 1.95 * np.power(this_voxel_preprocessed_ts_STD,2)
#         
#             # Continue
#             kalmThreshold[voxel] = 0.9 * this_voxel_preprocessed_ts_STD
#             #print(t,S)
#             [data_filtered,S[voxel],
#              fPositDerivSpike[voxel], fNegatDerivSpike[voxel]] = kalman_filter(kalmThreshold[voxel],Data_InMask_Preprocessed[voxel,t],S[voxel],fPositDerivSpike[voxel], fNegatDerivSpike[voxel])
#             #print(t,S)
#             Data_InMask_Filtered[voxel,t] = data_filtered

# +
Data_InMask_Step01 = np.zeros(Data_InMask.shape)
Data_InMask_Step02 = np.zeros(Data_InMask.shape)
Data_InMask_Step03 = np.zeros(Data_InMask.shape)
#Data_InMask_Preprocessed = np.zeros(Data_InMask.shape)
Regressor_Coeffs         = np.zeros((n_regressors,Data_Nv,Data_Nt))
#Data_InMask_Filtered = np.zeros(Data_InMask_Preprocessed.shape)

for t in tqdm(np.arange(Data_Nt)):
#for t in tqdm(np.arange(75)):
    # Initializations
    if t not in Vols4Preproc:
        continue
    if t == Vols4Preproc[0]:
        n                = 1 # Initialize counter
        prev_iGLM        = {} # Initialization for iGML
        S                = np.reshape(np.repeat({'Q':0, 'P':0, 'x':0, 'R':0},Data_Nv*Data_Nt), (Data_Nv,Data_Nt),order='F') # Initializations for Kalman Filter
        fPositDerivSpike = np.zeros((Data_Nv, Data_Nt))
        fNegatDerivSpike = np.zeros((Data_Nv, Data_Nt))
        kalmThreshold    = np.zeros((Data_Nv, Data_Nt))
    else:
        n = n +1;

    Yn = Data_InMask[:,t][:,np.newaxis]
    Fn = nuisance[t,:][:,np.newaxis]
    
    # 1) Remove Nuissance Regressors from data
    Yn_d,prev_iGLM, Bn      = rt_regress_vol(n,Yn,Fn,prev_iGLM)
    Data_InMask_Step01[:,t] = Yn_d
    Regressor_Coeffs[:,:,t] = Bn.T
    
    # 2) Low Pass Filter
    if t > (Vols4Preproc[0] + 1):
        for voxel in np.arange(Data_Nv):
            input_ts     = Data_InMask_Step01[voxel,np.arange(DONT_USE_VOLS,t+1)] #+1 needed to include this last volume due to arange finalizing at -1
            input_ts_STD = np.std(input_ts) 
            
            input_S= {'Q': 0.25*np.power(input_ts_STD,2),
                      'R': np.power(input_ts_STD,2),
                      'x': S[voxel,t-1]['x'],
                      'P': S[voxel,t-1]['P']}
            input_fPos = fPositDerivSpike[voxel,t-1]
            input_fNeg = fNegatDerivSpike[voxel,t-1]
            
            # Continue
            kalmThreshold[voxel,t] = 0.9 * input_ts_STD

            [data_filtered, out_S, out_fPos, out_fNeg] = kalman_filter(kalmThreshold[voxel,t],
                                                                       input_ts[-1],
                                                                       input_S,
                                                                       input_fPos,
                                                                       input_fNeg)
            Data_InMask_Step02[voxel,t] = data_filtered
            S[voxel,t] = out_S
            fPositDerivSpike[voxel,t] = out_fPos
            fNegatDerivSpike[voxel,t] = out_fNeg
    Yn_df = Data_InMask_Step02[:,t]
    
    # 3) Smooth Image
    Yn_d_vol    = unmask_fMRI_img(Yn_df, Mask_Img)
    Yn_d_vol_sm = smooth_array(Yn_d_vol,affine=Data_Img.affine, fwhm=FWHM)
    Yn_sm       = mask_fMRI_img(Yn_d_vol_sm, Mask_Img)
    
    # 4) Update Global Structures
    Data_InMask_Step03[:,t] = Yn_sm
# -

# ***
# ***
# ***

plt.figure(figsize=(20,5))
plt.plot(Data_InMask_Step01[voxel-1000,np.arange(40,500)])
plt.plot(Data_InMask_Step02[voxel-1000,np.arange(40,500)])
plt.plot(Data_InMask_Step03[voxel-1000,np.arange(40,500)])


def kalman_filter_mv(kalmTh, kalmIn, S, fPositDerivSpike, fNegatDerivSpike):
    # Preset
    Nv = kalmIn.shape[0]
    A = np.ones(Nv,1)
    H = np.ones(Nv,1)
    I = np.ones(Nv,1)


# ***
# ***
# ***

out = unmask_fMRI_img(Data_InMask_Step01, Mask_Img, '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/Test_Step01.nii')
out = unmask_fMRI_img(Data_InMask_Step02, Mask_Img, '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/Test_Step02.nii')
out = unmask_fMRI_img(Data_InMask_Step03, Mask_Img, '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/Test_Step03.nii')
for i,lab in enumerate(nuisance_labels):
    data = Regressor_Coeffs[i,:,:]
    out = unmask_fMRI_img(data, Mask_Img, '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/Test_'+lab+'.nii')

# ***
# # Data Standarization in Space

# Standarize CAPs in Space
# ========================
sc_CAPs = StandardScaler(with_mean=True, with_std=True)
CAPs_InMask = sc_CAPs.fit_transform(CAPs_InMask)

# Standarize Time series in Space
# ===============================
sc_Data_space  = StandardScaler(with_mean=True, with_std=True)
Data_InMask_dn = sc_Data_space.fit_transform(Data_InMask_Step03)

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
LinReg_Results[CAP_labels].hvplot(legend='top', label='Regression Coefficients for all CAPs').opts(width=1500)).cols(1)
hv.save(linear_regression_layout,'./LinearRegression_Results.png',backend='matplotlib')
linear_regression_layout



















TRAIN_InMask_d = np.zeros(TRAIN_InMask.shape)
for i,v in tqdm(enumerate(Vols4Detrending), total=Vols4Detrending.shape[0]):
    # This Acq Data + Nuisance
    n = i + 1
    Yn = TRAIN_InMask[:,v][:,np.newaxis]
    Fn = nuisance[v,:][:,np.newaxis]
    
    # Initializations for iGLM (only happens once)
    if v == Vols4Detrending[0]:
        L   = Fn.shape[0]
        Cn  = np.zeros((L,L), dtype='float64')
        Dn  = np.zeros((TRAIN_InMask_Nv, L), dtype='float64')
        s2n = np.zeros((TRAIN_InMask_Nv, 1), dtype='float64')
    
    # Detrend Volume
    Bn,Cn,Dn,s2n = iGLMVol(n,Yn,Fn,Dn,Cn,s2n)
    Yn_d = Yn - np.matmul(Bn,Fn)
    TRAIN_InMask_d[:,v] = np.squeeze(Yn_d)

Vols4Preproc[0]

















TRAIN_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/FeedingData/TrainingData.volreg.nii'
Motion_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/FeedingData/TrainingData.Motion.1D'
MASK_Path         = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/RTtest/GMribbon_R4Feed.nii'
CAPs_Path         = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/RTtest/Frontier2013_CAPs_R4Feed.nii'
SVRs_Path  = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/D00_OriginalData/SVRs.pkl'

TRAIN_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/TECH07_Run01_Training.nii'
Motion_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/TECH07_Run01_Training.Motion.1D'
MASK_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/rt.__002.mask.FB.nii'
CAPs_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/Frontier2013_CAPs_R4Feed.nii'
SVRs_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/Offline_SVRs.pkl'

DONT_USE_VOLS   = 10
FIRST_VALID_VOL = 50

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import holoviews as hv
import hvplot.pandas
from scipy.stats import zscore
from sklearn.svm import SVR
from scipy.special import legendre
from scipy.signal  import detrend
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numpy.linalg import cholesky, inv, pinv
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
#TRAIN_InMask = TRAIN_InMask[:,np.arange(DONT_USE_VOLS,TRAIN_Nt)] # Discard Volumes not yet stable (takes some time for on-line operations to stabilize)
[TRAIN_InMask_Nv, TRAIN_InMask_Nt] = TRAIN_InMask.shape
print('++ Final CAPs Template Dimensions: %d CAPs with %d voxels' % (CAPs_InMask.shape[1],CAPs_InMask.shape[0]))
print('++ Final Training Data Dimensions: %d Acqs with %d voxels' % (TRAIN_InMask_Nt, TRAIN_InMask_Nv))
# -

# ***
# # Detrend Data (as if it happens online)

Motion = np.loadtxt(Motion_Path) #[np.arange(DONT_USE_VOLS,TRAIN_Nt),:]
nuisance_motion = Motion - Motion.mean(axis=0)
print('++ Motion Regressor Dimensions [%s]' % str(Motion.shape))

# Create Polort
polort          = 2
min             = -1.0
max             = 1.0
vols_to_discard = 10
#nuisance_polort = np.zeros((TRAIN_Nt-vols_to_discard, polort))
nuisance_polort = np.zeros((TRAIN_Nt, polort))
for n in range(polort):
    Pn = legendre(n)
    x = np.linspace(-1,1,TRAIN_Nt)
    nuisance_polort[:,n] = Pn(x).T
print('++ Polort Regressor Dimensions [%s]' % str(nuisance_polort.shape))

nuisance = np.concatenate((nuisance_polort,nuisance_motion),axis=1)
nuisance_DF = pd.DataFrame(nuisance,columns=['Polort'+str(i) for i in np.arange(polort)] + ['roll','pitch','yaw','dS','dL','dP'])
nuisance_DF.hvplot()

Vols4Detrending = np.arange(DONT_USE_VOLS,TRAIN_InMask_Nt)
Vols4Detrending.shape

TRAIN_InMask_d = np.zeros(TRAIN_InMask.shape)
for i,v in tqdm(enumerate(Vols4Detrending), total=Vols4Detrending.shape[0]):
    # This Acq Data + Nuisance
    n = i + 1
    Yn = TRAIN_InMask[:,v][:,np.newaxis]
    Fn = nuisance[v,:][:,np.newaxis]
    
    # Initializations for iGLM (only happens once)
    if v == Vols4Detrending[0]:
        L   = Fn.shape[0]
        Cn  = np.zeros((L,L), dtype='float64')
        Dn  = np.zeros((TRAIN_InMask_Nv, L), dtype='float64')
        s2n = np.zeros((TRAIN_InMask_Nv, 1), dtype='float64')
    
    # Detrend Volume
    Bn,Cn,Dn,s2n = iGLMVol(n,Yn,Fn,Dn,Cn,s2n)
    Yn_d = Yn - np.matmul(Bn,Fn)
    TRAIN_InMask_d[:,v] = np.squeeze(Yn_d)

output = np.zeros((MASK_Nx*MASK_Ny*MASK_Nz,TRAIN_InMask_d.shape[1]))
output[MASK_Vector==1,:] = TRAIN_InMask_d
output = np.reshape(output,(MASK_Nx,MASK_Ny,MASK_Nz,TRAIN_InMask_d.shape[1]),order='F')
output_img = nib.Nifti1Image(output,affine=MASK_Img.affine)
output_img.to_filename('/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/TECH07_Run01_TrainingData.detrended.nii')

from nilearn.image import smooth_img

output_img_sm = smooth_img(output_img,6)

output_img_sm.to_filename('/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/TECH07_Run01_TrainingData.detrended.smooth.nii')

SMOOTH_Vector = np.reshape(output_img_sm.get_data(),(TRAIN_Nx*TRAIN_Ny*TRAIN_Nz, TRAIN_Nt), order='F')
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
sc_TRAIN_space     = StandardScaler(with_mean=True, with_std=True)
# JAVIER TRAIN_InMask_dn    = sc_TRAIN_space.fit_transform(TRAIN_InMask_d)
TRAIN_InMask_dn    = sc_TRAIN_space.fit_transform(SMOOTH_InMask)

output = np.zeros((MASK_Nx*MASK_Ny*MASK_Nz,TRAIN_InMask_d.shape[1]))
output[MASK_Vector==1,:] = TRAIN_InMask_dn
output = np.reshape(output,(MASK_Nx,MASK_Ny,MASK_Nz,TRAIN_InMask_d.shape[1]),order='F')
output_img = nib.Nifti1Image(output,affine=MASK_Img.affine)
output_img.to_filename('/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/TECH07_Run01_TrainingData.detrended.norm.nii')

# ***
# # Generation of Traning Labels (via Linear Regression + Z-scoring)

Vols4Training = np.arange(FIRST_VALID_VOL,TRAIN_InMask_Nt)
Vols4Training.shape

# Initialize a Dictionary for results
Results = {}
for cap in CAP_labels:
    Results[cap]=[]
Results['R2'] = []

X_fmri = pd.DataFrame(CAPs_InMask, columns=CAP_labels)
for vol in tqdm(Vols4Training):
    Y_fmri = pd.Series(TRAIN_InMask_dn[:,vol],name='V'+str(vol).zfill(4))
    lm     = linear_model.LinearRegression()
    model  = lm.fit(X_fmri,Y_fmri)
    for i,cap in enumerate(CAP_labels):
        Results[cap].append(lm.coef_[i])
    Results['R2'].append(lm.score(X_fmri,Y_fmri))
LinReg_Results       = pd.DataFrame(Results)
LinReg_Results['TR'] = Vols4Training

linear_regression_layout = (hv.Curve(LinReg_Results, kdims=['TR'],vdims=['R2'], label='R2 for Linear Regression on Training Data').opts(width=1500, tools=['hover']) + \
LinReg_Results.drop(['TR','R2'], axis=1).hvplot(legend='top', label='Regression Coefficients for all CAPs').opts(width=1500)).cols(1)
hv.save(linear_regression_layout,'./LinearRegression_Results.png',backend='matplotlib')
linear_regression_layout

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
    mySVR.fit(TRAIN_InMask_dn[:,Vols4Training].T,Training_Labels)
    SVRs[cap_lab] = mySVR

import pickle
pickle_out = open(SVRs_Path,"wb")
pickle.dump(SVRs, pickle_out)
pickle_out.close()













plt.plot(TRAIN_InMask[1000,np.arange(100,490)])

plt.plot(TRAIN_InMask_d[1000,np.arange(100,490)])

output = np.zeros((MASK_Nx*MASK_Ny*MASK_Nz,TRAIN_InMask_d.shape[1]))
output[MASK_Vector==1,:] = TRAIN_InMask_d
output = np.reshape(output,(MASK_Nx,MASK_Ny,MASK_Nz,TRAIN_InMask_d.shape[1]),order='F')
output_img = nib.Nifti1Image(output,affine=MASK_Img.affine)
output_img.to_filename('/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/TrainingData.detrended.nii')


# K = Nv = Number of Voxels
# L = Number of nuisance regressors
def iGLMVol(n,Yn,Fn,Dn,Cn,s2n):
    """ 
    n:int 
        Current acq number 
    Yn:array[Nv,1] 
        Voxel-wise data at point n
    Dn:array[Nv,L]
        Previous estimation of Dn. It will be updated here
    Fn:array[L,1]
        Nuisance Regressors values at point n
    Cn:array[]
        Matrix for Cholesky Decomposition
    """
    
    
    nrBasFct = Fn.shape[0]                   # Number of Regressors
    Bn = np.zeros((nrBasFct,1))
    df = n - nrBasFct                        # Degrees of Freedom
    
    Dn = Dn + (Yn*Fn.T)                      # Eq. 17
    Cn = (((n-1)/n) * Cn) + ((1/n)*Fn*Fn.T)  # Eq. 18
    s2n = s2n + (Yn * Yn)            # Eq. 9 without the 1/n factor... see below
    Nn = cholesky(Cn).T # Matlab difference: https://docs.scipy.org/doc/numpy-1.13.0/user/numpy-for-matlab-users.html
    # Auxiliary Coeffs, Original Coeffs, and Error Term estimation
    if n > nrBasFct + 2:
        invNn = inv(Nn.T)
        An    = (1/n) * np.matmul(Dn,invNn.T)  # Eq. 14
        Bn    = np.matmul(An,invNn)            # Eq. 16
        e2n   = (n/df) * ( (s2n/n) - np.sum(An*An,axis=1) )  # Eq. 8,9, 22
        # Handle negative e2n
        [neg_e2n,_] = np.where(e2n<0.0)
        if neg_e2n.size != 0:
            e2n[neg_e2n] = np.abs(e2n[neg_e2n])
        # Handle zero e2n
        [zero_e2n,_] = np.where(e2n==0.0)
        if neg_e2n.size != 0:
            e2n[zero_e2n] = 1e-10 #In openNFT was 1e10 (Does not make sense)
    return Bn,Cn,Dn,s2n    

np.all(np.linalg.eigvals(Cn) > 0)

n =1 
Yn = TRAIN_InMask[:,n][:,np.newaxis]
Fn = nuisance[n,:][:,np.newaxis]

np.matmul(Fn,Fn.T)

cholesky(Cn)

L = 9
Cn  = np.zeros((L,L))
Dn  = np.zeros((TRAIN_InMask_Nv, L))
s2n = np.zeros((TRAIN_InMask_Nv, 1))
Bn  = np.zeros((TRAIN_InMask_Nt,L))
n =1 
Yn = TRAIN_InMask[:,n][:,np.newaxis]
Fn = nuisance[n,:][:,np.newaxis]

np.arange(1,TRAIN_InMask_Nt+1).shape

iGLMVol(n,Yn,Fn,Dn,Cn,s2n)

n =1 
Yn = TRAIN_InMask[:,n][:,np.newaxis]
Fn = nuisance[n,:][:,np.newaxis]

# +
nrBasFct = Fn.shape[0]  
Bn = np.zeros((nrBasFct,1))
df = n - nrBasFct                        # Degrees of Freedom
    
Dn = Dn + np.matmul(Yn,Fn.T)             # Eq. 17
Cn = (((n-1)/n) * Cn) + ((1/n)*np.matmul(Fn,Fn.T))  # Eq. 18
s2n = s2n + (Yn * Yn)  
# -

Cn

Dn = Dn + np.matmul(Yn,Fn.T) 
Cn = (((n-1)/n) * Cn) + ((1/n)*np.matmul(Fn,Fn.T))

Cn

polort = 3
min    = -1.0
max    = 1.0
vols_to_discard = 100
num_regerssors  = 6 + polort
RESULT = np.zeros(TRAIN_InMask.shape)
for vol in tqdm(np.arange(0,200)):
    if vol < vols_to_discard+num_regerssors:
        continue
    # Create Nuisance Polort
    nuisance_polort = np.zeros((vol-vols_to_discard, polort))
    for n in range(polort):
        Pn = legendre(n)
        x = np.linspace(-1,1,vol-vols_to_discard)
        nuisance_polort[:,n] = Pn(x).T
    # Create Nuisance Motion
    nuisance_motion = Motion[np.arange(vols_to_discard,vol),:]
    nuisance_motion = nuisance_motion - nuisance_motion.mean(axis=0)
    # Concatenate All Regressors
    nuisance = np.concatenate((nuisance_polort,nuisance_motion),axis=1)
    nuisance = pd.DataFrame(nuisance,columns=['Polort0','Polort1','Polort2','roll','pitch','yaw','dS','dL','dP'])
    
    # Data
    data = pd.Series(voxel[np.arange(vols_to_discard,vol)])
    
    # Clean
    for v in np.arange(TRAIN_InMask_Nv):
        data = pd.Series(TRAIN_InMask[v,np.arange(vols_to_discard,vol)])
        mod = sm.RecursiveLS(data, nuisance)
        res = mod.fit()
        RESULT[v,vol] = res.resid.tail(1).values[0]

res.resid.tail(1).values[0]

# +
mod = sm.RecursiveLS(data, nuisance)
res = mod.fit()

print(res.summary())
# -

#print(res.recursive_coefficients.filtered[0])
res.plot_recursive_coefficient(range(mod.k_exog), alpha=None, figsize=(10,30));

plt.plot(res.resid)

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
