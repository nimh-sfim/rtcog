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

TRAIN_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/FeedingData/TrainingData.volreg.nii'
Motion_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/FeedingData/TrainingData.Motion.1D'
MASK_Path         = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/RTtest/GMribbon_R4Feed.nii'
CAPs_Path         = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/RTtest/Frontier2013_CAPs_R4Feed.nii'
SVRs_Path  = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/D00_OriginalData/SVRs.pkl'
DONT_USE_VOLS   = 10
FIRST_VALID_VOL = 100

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

Motion = np.loadtxt(Motion_Path)[np.arange(DONT_USE_VOLS,TRAIN_Nt),:]
nuisance_motion = Motion - Motion.mean(axis=0)
print(Motion.shape)

# Create Polort
polort          = 3
min             = -1.0
max             = 1.0
vols_to_discard = 10
nuisance_polort = np.zeros((TRAIN_Nt-vols_to_discard, polort))
for n in range(polort):
    Pn = legendre(n)
    x = np.linspace(-1,1,TRAIN_Nt-vols_to_discard)
    nuisance_polort[:,n] = Pn(x).T

nuisance = np.concatenate((nuisance_polort,nuisance_motion),axis=1)
nuisance_DF = pd.DataFrame(nuisance,columns=['Polort0','Polort1','Polort2','roll','pitch','yaw','dS','dL','dP'])
nuisance_DF.hvplot()

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
TRAIN_InMask = TRAIN_InMask[:,np.arange(DONT_USE_VOLS,TRAIN_Nt)] # Discard Volumes not yet stable (takes some time for on-line operations to stabilize)
[TRAIN_InMask_Nv, TRAIN_InMask_Nt] = TRAIN_InMask.shape
print('++ Final CAPs Template Dimensions: %d CAPs with %d voxels' % (CAPs_InMask.shape[1],CAPs_InMask.shape[0]))
print('++ Final Training Data Dimensions: %d Acqs with %d voxels' % (TRAIN_InMask_Nt, TRAIN_InMask_Nv))


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
        #e2n   = (n/df) * ( (s2n/n) - np.sum(An*An,axis=1) )  # Eq. 8,9, 22
        ## Handle negative e2n
        #[neg_e2n,_] = np.where(e2n<0.0)
        #if neg_e2n.size != 0:
        #    e2n[neg_e2n] = np.abs(e2n[neg_e2n])
        ## Handle zero e2n
        #[zero_e2n,_] = np.where(e2n==0.0)
        #if neg_e2n.size != 0:
        #    e2n[zero_e2n] = 1e-10 #In openNFT was 1e10 (Does not make sense)
    else:
        print ('%d non positive definite'% n)
        Bn = np.zeros((nv,nrBasFct))
    return Bn,Cn,Dn,s2n


# -

TRAIN_InMask_d = np.zeros(TRAIN_InMask.shape)
for v in tqdm(np.arange(TRAIN_InMask_Nt)):
    n = v + 1
    Yn = TRAIN_InMask[:,v][:,np.newaxis]
    Fn = nuisance[v,:][:,np.newaxis]
    if n == 1:
        L   = Fn.shape[0]
        Cn  = np.zeros((L,L), dtype='float64')
        Dn  = np.zeros((TRAIN_InMask_Nv, L), dtype='float64')
        s2n = np.zeros((TRAIN_InMask_Nv, 1), dtype='float64')
    Bn,Cn,Dn,s2n = iGLMVol(n,Yn,Fn,Dn,Cn,s2n)
    Yn_d = Yn - np.matmul(Bn,Fn)
    TRAIN_InMask_d[:,v] = np.squeeze(Yn_d)

plt.plot(TRAIN_InMask[1000,np.arange(20,490)])

plt.plot(TRAIN_InMask_d[1000,np.arange(20,490)])

output = np.zeros((MASK_Nx*MASK_Ny*MASK_Nz,TRAIN_InMask_d.shape[1]))
output[MASK_Vector==1,:] = TRAIN_InMask_d
output = np.reshape(output,(MASK_Nx,MASK_Ny,MASK_Nz,TRAIN_InMask_d.shape[1]),order='F')
output_img = nib.Nifti1Image(output,affine=MASK_Img.affine)
output_img.to_filename('/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH06/FeedingData/TrainingData.volreg.detrended.nii')


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
