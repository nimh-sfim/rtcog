import sys
import os.path as osp
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import MaxAbsScaler
from scipy.stats import zscore
from sklearn.svm import SVR
from tqdm import tqdm
import pickle
import csv
import matplotlib.pyplot as plt
import holoviews as hv
import hvplot.pandas
import multiprocessing as mp
import panel as pn
from bokeh.palettes import Category10_7
tqdm().pandas()

from rtcog.utils.fMRI import load_fMRI_file, mask_fMRI_img

import logging
log     = logging.getLogger("training")
log_fmt = logging.Formatter('[%(levelname)s - training]: %(message)s')
log_ch  = logging.StreamHandler()
log_ch.setFormatter(log_fmt)
log.setLevel(logging.INFO)
log.addHandler(log_ch)

def train_one_svr(input_dict):
    template_label = input_dict['template_label']
    print(' +++ Starting for %s' % template_label)
    data    = input_dict['data']
    labels  = input_dict['labels']
    C       = input_dict['C']
    epsilon = input_dict['epsilon']
    svr = SVR(kernel='linear', C=C, epsilon=epsilon)
    svr.fit(data,labels)
    print(' --- Ending for %s' % template_label)
    return svr

class SVRtrainer(object):
    def __init__(self, opts):
        self.C         = 1.0
        self.epsilon   = 0.01
        self.SVRs      = {}
        self.mask_path = opts.mask_path
        self.data_path = opts.data_path
        self.nvols_discard = opts.nvols_discard
        self.do_lasso      = not(opts.no_lasso)
        self.lasso_alpha   = opts.lasso_alpha
        self.lasso_pos     = not(opts.lasso_no_pos_only) 

        # Ensure outdir points to a file within an existing dir.
        if not osp.isdir(opts.outdir):
            raise ValueError('Output directory does not exist.')
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

        # Ensure templates file exists
        if not osp.exists(opts.templates_path):
            log.error('templates Template does not exists. Please correct.')
            sys.exit(-1)
        
        self.templates_path = opts.templates_path
        self.template_labels_path = opts.template_labels_path
    
    def load_datasets(self):
        try:
            with open (self.template_labels_path, 'r') as f:
                lines = f.read()
                self.template_labels = [label.strip() for label in lines.strip().split(',')]
        except FileNotFoundError:
            log.error('Template labels file does not exist.')
            sys.exit(-1)
        except Exception as e:
            log.error(e)
            sys.exit(-1)

        self.templates_img = load_fMRI_file(self.templates_path, verbose=True)
        self.mask_img = load_fMRI_file(self.mask_path, verbose=True)
        self.data_img = load_fMRI_file(self.data_path, verbose=True)

        self.templates_masked = mask_fMRI_img(self.templates_img, self.mask_img)
        # Now assuming we are using every template within a file.
        # self.templates_masked = self.templates_masked[:, self.templates_indexes]  # Select only templates of interest 
        self.data_masked = mask_fMRI_img(self.data_img, self.mask_img)
        [self.data_nv, self.data_nt] = self.data_masked.shape
        log.debug('Masked templates Dimensions: %s' % str(self.templates_masked.shape))
        log.debug('Masked Data Dimensions: %s' % str(self.data_masked.shape))
    
    def generate_training_labels(self):
        self.vols4training = np.arange(self.nvols_discard, self.data_nt)
        log.debug('[generate_training_labels] Number of volumes for training [%d]: [min=%d, max=%d] (Python)' % (self.vols4training.shape[0], np.min(self.vols4training), np.max(self.vols4training)))
        
        # Initialize Results Structure
        results = {}
        for template in self.template_labels:
            results[template] = []
        self.lm_R2 = []

        # Perform linear regression 
        X_fmri = pd.DataFrame(self.templates_masked, columns=self.template_labels)
        #for vol in tqdm(self.vols4training):
        for vol in tqdm(range(self.data_nt)):
            if vol in self.vols4training:
                Y_fmri = pd.Series(self.data_masked[:,vol], name='V'+str(vol).zfill(4))
                if self.do_lasso:
                    lm = linear_model.Lasso(alpha=self.lasso_alpha, max_iter=5000, positive=self.lasso_pos)
                else:
                    lm = linear_model.LinearRegression()
                model  = lm.fit(X_fmri, Y_fmri)
                for i, template in enumerate(self.template_labels):
                    results[template].append(lm.coef_[i])
                self.lm_R2.append(lm.score(X_fmri, Y_fmri))
            else:
                for i, template in enumerate(self.template_labels):
                    results[template].append(0)
                self.lm_R2.append(0)

        lm_results = pd.DataFrame(results)
        self.LR_labels_preZscore = lm_results

        # Z-score the results from the linear regression
        [lmr_nt, lmr_ntemplates] = lm_results.shape
        
        if self.do_lasso:
            mas               = MaxAbsScaler()
            lm_results_flat_Z = mas.fit_transform(lm_results)
        else:
            lm_results_flat   = lm_results
            lm_results_flat   = lm_results_flat.values.reshape(lmr_ntemplates*lmr_nt, order='F')
            lm_results_flat_Z = zscore(lm_results_flat)
            lm_results_flat_Z = lm_results_flat_Z.reshape((lmr_nt,lmr_ntemplates), order='F')

        self.lm_res_z = pd.DataFrame(lm_results_flat_Z, columns=self.template_labels, index=lm_results.index)

    def train_svrs(self):
        for template_lab in tqdm(self.template_labels):
            Training_Labels = self.lm_res_z[template_lab]
            mySVR = SVR(kernel='linear', C=self.C, epsilon=self.epsilon)
            mySVR.fit(self.data_masked[:,self.vols4training].T,Training_Labels[self.vols4training])
            self.SVRs[template_lab] = mySVR
        return 1
    
    def train_svrs_mp(self):
        num_cores = len(self.template_labels)
        pool = mp.Pool(num_cores)  # Create as many processes as SVRs need to be trained
        inputs = ({'template_label': template_lab,
                   'data'   : self.data_masked[:,self.vols4training].T,
                   'labels' : self.lm_res_z[template_lab][self.vols4training],
                   'C'      : self.C,
                   'epsilon': self.epsilon} for template_lab in self.template_labels)
        log.info(' +++ About to go parallel with %d cores' % (num_cores))       
        res = pool.map(train_one_svr, inputs)
        log.info(' +++ All parallel operations completed.')
        self.SVRs = {template_lab:res[c] for c,template_lab in enumerate(self.template_labels)}
        
        # Shut down pool 
        pool.close()
        pool.join()

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
        plt.plot(np.arange(self.data_nt),self.lm_R2)
        plt.ylabel('R2 (Linear Regression)')
        plt.subplot(212)
        plt.plot(self.lm_res_z)
        plt.xlabel('Time [TRs]')
        plt.ylabel('Z-Score')
        plt.legend(self.template_labels)
        plt.tight_layout()
        plt.savefig(self.outpng, dpi=200)
        log.info(' - save_results - Saved Label Computation Results to [%s]' % self.outpng)

        # Save Dynamic Figures
        #renderer    = hv.renderer('bokeh')
        R2_DF       = pd.DataFrame(columns=['TR','R2'], index=np.arange(self.data_nt))
        R2_DF['TR'] = R2_DF.index
        R2_DF['R2'] = self.lm_R2
        R2_curve    = R2_DF.hvplot(legend='top', label='R2 for Linear Regression on Training Data', x='TR').opts(width=1500, height=200)

        Labels_DF        = self.LR_labels_preZscore.copy()
        Labels_DF['TR']  = Labels_DF.index
        Labels_curve     = Labels_DF.hvplot(legend='top', label='Labels for SVR Training (pre Z-scoring)', x='TR').opts(width=1500, height=200)

        ZLabels_DF       = self.lm_res_z.copy()
        ZLabels_DF['TR'] = ZLabels_DF.index
        ZL_curve         = ZLabels_DF.hvplot(legend='top', label='Labels for SVR Training (post Z-scoring)', x='TR').opts(width=1500, height=200)

        for i,(template,color) in enumerate(zip(self.template_labels,Category10_7)):
            if i == 0:
                Hist_Layout = ZLabels_DF.hvplot.hist(y=template, fill_color=color, width=500, height=200)
            else:
                Hist_Layout = Hist_Layout + ZLabels_DF.hvplot.hist(y=template, fill_color=color,  width=500, height=200)

        Hist_Layout.cols(3)
        LM_Layout        = (R2_curve + Labels_curve + ZL_curve).cols(1).opts(shared_axes=False)

        pn.Column(LM_Layout,Hist_Layout).save(self.outhtml)
        #renderer.save(LM_Layout, self.outhtml)
        log.info(' - save_results - Saved Label Computation Results (Dynamic View) to [%s.html]' % self.outhtml)
        return 1
    