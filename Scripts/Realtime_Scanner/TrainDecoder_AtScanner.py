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


import sys
from   argparse import ArgumentParser,RawTextHelpFormatter
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import holoviews as hv
import hvplot.pandas
from scipy.stats import zscore
from sklearn.svm import SVR
import pickle
hv.extension('bokeh')


from tqdm import tqdm
tqdm().pandas()

#TRAIN_Path = None
#MASK_Path = './GMribbon_R4Feed.nii'
#CAPs_Path = './Frontier2013_CAPs_R4Feed.nii'
#SVRs_Path = './Online_SVRs.pkl'
#TRAIN_NVols_Discard = 90

# Load CAPs in same grid as functional (linearly interpolated)
# ============================================================
def load_CAPs(CAPs_Path):
    CAPs_Img  = nib.load(CAPs_Path)
    [CAPs_Nx, CAPs_Ny, CAPs_Nz, n_CAPs] = CAPs_Img.shape
    print (' + CAPs Dimensions = %d CAPs | [%d,%d,%d]' % (n_CAPs, CAPs_Nx,CAPs_Ny,CAPs_Nz))

    return CAPs_Img, CAPs_Nx, CAPs_Ny, CAPs_Nz, n_CAPs

# Load FOV mask in same grid as functional (ensure it only cover areas that have data on input and CAPs)
# ======================================================================================================
def load_MASK(MASK_Path):
    MASK_Img  = nib.load(MASK_Path)
    [MASK_Nx, MASK_Ny, MASK_Nz] = MASK_Img.shape
    MASK_Nv = MASK_Img.get_data().sum()
    print (' + Mask Dimensions = %d Voxels in mask | [%d,%d,%d]' % (MASK_Nv, MASK_Nx,MASK_Ny,MASK_Nz))
    print (' + Mask Path: %s' % MASK_Path)

    return MASK_Img,MASK_Nx,MASK_Ny,MASK_Nz,MASK_Nv

# Load Training Dataset
# =====================
def load_train_data(TRAIN_Path):
    TRAIN_Img = nib.load(TRAIN_Path)
    [TRAIN_Nx, TRAIN_Ny, TRAIN_Nz, TRAIN_Nt] = TRAIN_Img.shape
    print (' + Data Dimensions = [%d,%d,%d, Nvols = %d]' % (TRAIN_Nx, TRAIN_Ny, TRAIN_Nz, TRAIN_Nt))

    return TRAIN_Img, TRAIN_Nx, TRAIN_Ny, TRAIN_Nz, TRAIN_Nt

def spatial_normalize(data):
    sc = StandardScaler(with_mean=True, with_std=True)
    norm_data = sc.fit_transform(data)
    return norm_data



def define_ArgParser ():
    """
       Process command line options for on-going experiment.
       Customize as needed for your own experiments.
    """
    
    description = "Decoder Training for rtCAPs project."
    parser = ArgumentParser(description=description,formatter_class=RawTextHelpFormatter)

    parser.add_argument('-d','--train_data_path', help='Path to the fMRI training dataset', 
                     type=str, dest='TRAIN_Path', default=None)

    parser.add_argument('-m','--mask_path', help='Path to the mask dataset', 
                     type=str, dest='MASK_Path', default='./GMribbon_R4Feed.nii')

    parser.add_argument('-c','--caps_path', help='Path to the CAPs dataset', 
                     type=str, dest='CAPs_Path', default='./Frontier2013_CAPs_R4Feed.nii')

    parser.add_argument('-o','--out_svrs', help='Path to output file with SVR objects', 
                     type=str, dest='SVRs_Path', default='./Online_SVRs.pkl')

    parser.add_argument('-D','--num_vols_discard', help='Number of vols to discard from training set', 
                     type=int, dest='TRAIN_NVols_Discard', default=90)

    return parser
    
if __name__ == '__main__':
    print('++ =================================================================')
    # 1) Read Input Parameters: port, fullscreen, etc..
    parser = define_ArgParser()
    opts = parser.parse_args()

    print('++ INPUT OPTIONS:')
    print(' + CAPs_Path    = %s' % str(opts.CAPs_Path))
    print(' + MASK_Path    = %s' % str(opts.MASK_Path))
    print(' + TRAIN_Path   = %s' % str(opts.TRAIN_Path))
    print(' + OUT_Path     = %s' % str(opts.SVRs_Path))
    print(' + Vols2Discard = %d' % opts.TRAIN_NVols_Discard)
    print('++ =================================================================')
    print('++ LOADING ALL DATA:')
    
    # 2) Load CAPs
    CAPs_Img, CAPs_Nx, CAPs_Ny, CAPs_Nz, n_CAPs = load_CAPs(opts.CAPs_Path)
    
    # 3) Generate CAPs Labels
    CAP_indexes = np.arange(n_CAPs)
    CAP_labels  = np.array(['CAP'+str(cap).zfill(2) for cap in CAP_indexes])
    rCAP_indexes = np.array([15,25,2,4,18,28,24,11,21])
    rCAP_labels  = np.array(['VMed','VPol','VLat','DMN','SMot','Audi','ExCn','rFPa','lFPa'])
    for i,ii in enumerate(rCAP_indexes):
        CAP_labels[ii] = rCAP_labels[i]

    # 4) Load Mask
    MASK_Img,MASK_Nx,MASK_Ny,MASK_Nz,MASK_Nv = load_MASK(opts.MASK_Path)

    # 5) Load Training Dataset
    TRAIN_Img, TRAIN_Nx, TRAIN_Ny, TRAIN_Nz, TRAIN_Nt = load_train_data(opts.TRAIN_Path)

    # 6) Vectorize All
    print('++ =================================================================')
    print('++ VECTORIING, NORMALIZING ALL DATA:')
    MASK_Vector  = np.reshape(MASK_Img.get_data(),(MASK_Nx*MASK_Ny*MASK_Nz),          order='F')
    CAPs_Vector  = np.reshape(CAPs_Img.get_data(),(CAPs_Nx*CAPs_Ny*CAPs_Nz, n_CAPs),  order='F')
    TRAIN_Vector = np.reshape(TRAIN_Img.get_data(),(TRAIN_Nx*TRAIN_Ny*TRAIN_Nz, TRAIN_Nt), order='F')

    # 7) Minimize rounding errors
    CAPs_Vector  = CAPs_Vector.astype('float64')
    TRAIN_Vector = TRAIN_Vector.astype('float64')

    # 8) Mask All (and keep only CAPs of interest)
    CAPs_InMask = CAPs_Vector[MASK_Vector==1,:]
    CAPs_InMask = CAPs_InMask[:,CAP_indexes]
    TRAIN_InMask = TRAIN_Vector[MASK_Vector==1,:]
    TRAIN_InMask = TRAIN_InMask[:,np.arange(opts.TRAIN_NVols_Discard,TRAIN_Nt)] # Discard Volumes not yet stable (takes some time for on-line operations to stabilize)
    [TRAIN_InMask_Nv, TRAIN_InMask_Nt] = TRAIN_InMask.shape
    print(' + Final CAPs Template Dimensions: %d CAPs with %d voxels' % (CAPs_InMask.shape[1],CAPs_InMask.shape[0]))
    print(' + Final Training Data Dimensions: %d Acqs with %d voxels' % (TRAIN_InMask_Nt, TRAIN_InMask_Nv))

    # 9) Spatial Normalization
    CAPs_InMask = spatial_normalize(CAPs_InMask)
    TRAIN_InMask = spatial_normalize(TRAIN_InMask)
    print('++ =================================================================')
    print('++ LINEAR REGRESSION:')

    # 10) Initialize a Dictionary for results
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
    
    print('++ =================================================================')
    print('++ ZSCORE LINEAR REGRESSION:')
    LinReg_Zscores = pd.DataFrame(columns=LinReg_Results.columns, index=LinReg_Results.index)
    for col in LinReg_Results.columns:
        LinReg_Zscores[col] = zscore(LinReg_Results[col])
    
    print('++ =================================================================')
    print('++ TRAIN SUPPORT VECTOR MACHINES:')
    C       = 1.0
    epsilon = 0.01
    SVRs    = {}
    for cap_lab in tqdm(rCAP_labels):
        Training_Labels = LinReg_Zscores[cap_lab]
        mySVR = SVR(kernel='linear', C=C, epsilon=epsilon)
        mySVR.fit(TRAIN_InMask.T,Training_Labels)
        SVRs[cap_lab] = mySVR

    pickle_out = open(opts.SVRs_Path,"wb")
    pickle.dump(SVRs, pickle_out)
    pickle_out.close()
    print('++ =================================================================')
    print('++ PROGRAM ENDED SUCCESSFULLY')
    print('++ =================================================================')

# ----------------------------------------------------------------------------