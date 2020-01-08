import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import ntpath
import os.path as osp
import sys
import numpy as np
import pandas as pd
from sklearn import linear_model
from scipy.stats import zscore
from sklearn.svm import SVR
import pickle
import argparse
import matplotlib.pyplot as plt
import csv

import holoviews as hv
import hvplot.pandas

log     = logging.getLogger("trainSVRs")
log_fmt = logging.Formatter('[%(levelname)s - Main]: %(message)s')
log_ch  = logging.StreamHandler()
log_ch.setFormatter(log_fmt)
log.setLevel(logging.INFO)
log.addHandler(log_ch)


sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))

from tqdm import tqdm
tqdm().pandas()

from rtcap_lib.fMRI import load_fMRI_file, mask_fMRI_img
# -------------------------------------------------------------------------------------
class Program(object):
    def __init__(self, opts):
        self.C         = 1.0
        self.epsilon   = 0.01
        self.SVRs      = {}
        self.mask_path = opts.mask_path
        self.data_path = opts.data_path
        self.nvols_discard = opts.nvols_discard
        # Ensure outdir points to a file within an existing dir.
        if not osp.isdir(opts.outdir):
            log.error('Output directory does not exist. Please correct.')
            sys.exit(-1)
        self.outdir = opts.outdir
        self.prefix = opts.prefix
        self.outpkl     = osp.join(self.outdir,self.prefix+".pkl")
        self.outpng     = osp.join(self.outdir,self.prefix+".png")
        self.outhtml    = osp.join(self.outdir,self.prefix) 
        self.labels_csv = osp.join(self.outdir,self.prefix+"_lm_z_labels.csv")
        self.r2_csv     = osp.join(self.outdir,self.prefix+"_lm_R2.csv")
        self.tvols_csv  = osp.join(self.outdir,self.prefix+"_training_vols.csv")
        if osp.exists(self.outpkl):
            log.warning('Output File does exists. File will be overwritten.')

        # Ensure CAPs file exists
        if not osp.exists(opts.caps_path):
            log.error('CAPs Template does not exists. Please correct.')
            sys.exit(-1)
        
        self.caps_path   = opts.caps_path
        self.caps_indexes = [25,4,18,28,24,11,21]
        self.caps_labels  = ['VPol','DMN','SMot','Audi','ExCn','rFPa','lFPa']
    
    def load_datasets(self):
        self.caps_img = load_fMRI_file(self.caps_path, verbose=True)
        self.mask_img = load_fMRI_file(self.mask_path, verbose=True)
        self.data_img = load_fMRI_file(self.data_path, verbose=True)

        self.caps_masked = mask_fMRI_img(self.caps_img,self.mask_img)
        self.caps_masked = self.caps_masked[:, self.caps_indexes]  # Select only CAPs of intrest 
        self.data_masked = mask_fMRI_img(self.data_img, self.mask_img)
        [self.data_nv, self.data_nt] = self.data_masked.shape
        log.debug('Masked CAPs Dimensions: %s' % str(self.caps_masked.shape))
        log.debug('Masked Data Dimensions: %s' % str(self.data_masked.shape))
    
    def generate_training_labels(self):
        self.vols4training = np.arange(self.nvols_discard, self.data_nt)
        log.debug('Number of volumes for training [%d]: [min=%d, max=%d] (Python)' % (self.vols4training.shape[0], np.min(self.vols4training), np.max(self.vols4training)))

        # Initialize Results Structure
        results = {}
        for cap in self.caps_labels:
            results[cap] = []
        self.lm_R2 = []

        # Perform linear regression 
        X_fmri = pd.DataFrame(self.caps_masked, columns=self.caps_labels)
        for vol in tqdm(self.vols4training):
            Y_fmri = pd.Series(self.data_masked[:,vol], name='V'+str(vol).zfill(4))
            lm     = linear_model.LinearRegression()
            model  = lm.fit(X_fmri, Y_fmri)
            for i, cap in enumerate(self.caps_labels):
                results[cap].append(lm.coef_[i])
            #results['R2'].append(lm.score(X_fmri, Y_fmri))
            self.lm_R2.append(lm.score(X_fmri, Y_fmri))
        
        lm_results = pd.DataFrame(results)
        #lm_results['TR'] = self.vols4training

        # Z-score the results from the linear regression
        [lmr_nt, lmr_ncaps] = lm_results.shape
        lm_results_flat     = lm_results
        lm_results_flat     = lm_results_flat.values.reshape(lmr_ncaps*lmr_nt, order='F')
        lm_results_flat_Z   = zscore(lm_results_flat)
        lm_results_flat_Z   = lm_results_flat_Z.reshape((lmr_nt,lmr_ncaps), order='F')
        self.lm_res_z       = pd.DataFrame(lm_results_flat_Z, columns=self.caps_labels, index=lm_results.index)
        #self.lm_res_z['TR'] = self.vols4training
        

    def train_svrs(self):
        for cap_lab in tqdm(self.caps_labels):
            Training_Labels = self.lm_res_z[cap_lab]
            mySVR = SVR(kernel='linear', C=self.C, epsilon=self.epsilon)
            mySVR.fit(self.data_masked[:,self.vols4training].T,Training_Labels)
            self.SVRs[cap_lab] = mySVR
        return 1

    def save_results(self):
        # Save Trained Models
        pickle_out = open(self.outpkl,"wb")
        pickle.dump(self.SVRs, pickle_out)
        pickle_out.close()
        log.info(' - save_results - Trained SVR models saved to %s' % self.outpkl)
        
        # List of Training Volumes
        with open(self.tvols_csv, 'w') as f:
            writer = csv.writer(f)
            for val in self.vols4training:
                writer.writerow([val])
        log.info(' - save_results - List of training volumes saved to disk [%s]' % self.tvols_csv)
        
        # Save Labels & R2
        with open(self.r2_csv, 'w') as f:
            writer = csv.writer(f)
            for val in self.lm_R2:
                writer.writerow([val])
        log.info(' - save_results - R2 for linear model saved to disk [%s]' % self.r2_csv)
        
        self.lm_res_z.to_csv(self.labels_csv, header=True, sep=',', index=False)
        log.info(' - save_results - Z-score Labels saved to disk [%s]' % self.labels_csv)

        # Save Static Figure
        fig = plt.figure(figsize=(20,5))
        plt.subplot(211)
        plt.plot(self.vols4training,self.lm_R2)
        plt.ylabel('R2 (Linear Regression)')
        plt.subplot(212)
        plt.plot(self.lm_res_z)
        plt.xlabel('Time [TRs]')
        plt.ylabel('Z-Score')
        plt.legend(self.caps_labels)
        plt.savefig(self.outpng, dpi=200, layout='tight')
        log.info(' - save_results - Saved Label Computation Results to [%s]' % self.outpng)

        # Save Dynamic Figures
        renderer    = hv.renderer('bokeh')
        R2_DF       = pd.DataFrame(columns=['TR','R2'], index=np.arange(len(self.vols4training)))
        R2_DF['TR'] = self.vols4training
        R2_DF['R2'] = self.lm_R2
        R2_curve    = R2_DF.hvplot(legend='top', label='R2 for Linear Regression on Training Data', x='TR').opts(width=1500)

        ZLabels_DF       = self.lm_res_z.copy()
        ZLabels_DF['TR'] = self.vols4training
        ZL_curve         = ZLabels_DF.hvplot(legend='top', label='Regression Coefficients for all CAPs', x='TR').opts(width=1500)
        LM_Layout        = (R2_curve + ZL_curve).cols(1)
        renderer.save(LM_Layout, self.outhtml)
        log.info(' - save_results - Saved Label Computation Results (Dynamic View) to [%s.html]' % self.outhtml)
        return 1

def processProgramOptions (self, options=None):
    parser = argparse.ArgumentParser(description="Train SVRs for spatial template matching")
    parser_inopts = parser.add_argument_group('Input Options','Inputs to this program')
    parser_inopts.add_argument("-d","--data", action="store", type=str, dest="data_path", default=None, help="path to training dataset [Default: %(default)s]", required=True)
    parser_inopts.add_argument("-m","--mask", action="store", type=str, dest="mask_path", default=None, help="path to mask [Default: %(default)s]", required=True)
    parser_inopts.add_argument("-c","--caps", action="store", type=str, dest="caps_path", default=None, help="path to caps template [Default: ]", required=True)
    parser_inopts.add_argument("--discard",   action="store", type=int, dest="nvols_discard",   default=100,  help="number of volumes [Default: %(default)s]")
    parser_outopts = parser.add_argument_group('Output Options','wWere to save results')
    parser_outopts.add_argument("-o","--outdir",  action="store", type=str, dest="outdir",  default='./', help="output directory [Default: %(default)s]")
    parser_outopts.add_argument("-p","--prefix", action="store", type=str, dest="prefix", default="svr", help="prefix for output file [Default: %(default)s]")
    return parser.parse_args(options)  

def main():
    # 1) Read Input Parameters
    log.info('1) Reading Program Inputs...')
    opts = processProgramOptions(sys.argv)
    log.debug('User Options: %s' % str(opts))

    # 2) Ensure Inputs Correctness
    if opts.mask_path is None:
        log.error('Mask is missing. Please provide one.')
        sys.exit(-1)
    if opts.data_path is None:
        log.error('Training data is missing. Please provide one.')
        sys.exit(-1)
    if opts.outdir is None:
        log.error('Output directory is missing. Please provide one.')
        sys.exit(-1)
    if opts.caps_path is None:
        log.error('Path to CAPs templates is missing. Please provide one.')
        sys.exit(-1)
    
    # 3) Initialize Progrma Object
    log.info('2) Initializing Program Object...')
    program = Program(opts)

    # 4) Load Datasets into memory
    log.info('3) Load all data into memory...')
    program.load_datasets()

    # 5) Generate Training labels via Linear Regression + Z-scoring
    log.info('4) Generate traning labels...')
    program.generate_training_labels()

    # 5) Train the SVRs
    log.info('5) Train SVRs...')
    program.train_svrs()

    # 6) Save results to disk
    log.info('6) Save results to disk...')
    program.save_results()
    
    return 1
    
if __name__ == '__main__':
   sys.exit(main())