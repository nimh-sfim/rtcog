import logging as log
import ntpath
import os.path as osp
import sys
import numpy as np

from optparse import OptionParser

log.basicConfig(format='[%(levelname)s]: %(message)s', level=log.DEBUG)
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
        # Ensure out_path points to a file within an existing dir.
        if osp.isdir(opts.out_path):
            log.error('Output Path is a dir, not a filename. Please correct.')
            sys.exit(-1)
        out_dir, out_file = ntpath.split(opts.out_path)
        if not osp.exists(out_dir):
            log.error('Output Path does not exists. Please correct.')
            sys.exit(-1)
        self.out_dir  = out_dir
        self.out_file = out_file
        self.out_path = opts.out_path
        if osp.exists(opts.out_path):
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
        print(self.vols4training)
        log.debug('Number of volumes for training [%d]: [min=%d, max=%d] (Python)' % (self.vols4training.shape[0], np.min(self.vols4training), np.max(self.vols4training)))
        
        data = self.data_img.get_data()

        # Initialize Results Structure
        results = {}
        for cap in self.caps_labels:
            results[cap] = []
        results['R2'] = []

        # Perform linear regression 
        X_fmri = pd.DataFrame(self.caps_masked, columns=self.caps_labels)
        for vol in tqdm(self.vols4training):
            Y_fmri = pd.Series(self.data[:,vol], name='V'+str(vol).zfill(4))
            lm     = linear_model.LinearRegression()
            model  = lm.fit(X_fmri, Y_fmri)
            for i, cap in enumerate(self.cap_labels):
                results[cap].append(lm_coef_[i])
            results['R2'].append(lm.score(X_fmri, Y_fmri))
        
        lm_results = pd.DataFrame(results)
        lm_results['TR'] = self.vols4training

        # Z-score the results from the linear regression
        [LR_Nt, LR_Ncaps]    = LinReg_Results[CAP_labels].shape
        All_LinReg           = LinReg_Results[CAP_labels]
        All_LinReg           = All_LinReg.values.reshape(LR_Ncaps*LR_Nt, order='F')
        All_LinReg_Z         = zscore(All_LinReg)
        All_LinReg_Z         = All_LinReg_Z.reshape((LR_Nt,LR_Ncaps), order='F')
        LinReg_Zscores       = pd.DataFrame(All_LinReg_Z, columns=CAP_labels, index=LinReg_Results.index)
        LinReg_Zscores['TR'] = Vols4Training
        LinReg_Zscores.hvplot(legend='top', label='Z-scored training labels', x='TR').opts(width=1500)


    def train_svrs(self):
        for cap_lab in tqdm(self.caps_labels):
            Training_Labels = LinReg_Zscores[cap_lab]
            mySVR = SVR(kernel='linear', C=C, epsilon=epsilon)
            mySVR.fit(Data_norm[:,Vols4Training].T,Training_Labels)
            SVRs[cap_lab] = mySVR
        


def processProgramOptions(options):
    usage = "%prog [options]"
    description = "rtCAPs: This program train classifiers for the on-line detection of \
                   CAP configurations in fMRI data."

    parser = OptionParser(usage = usage, description = description)
    parser.add_option("-d","--data", action="store", type="str", dest="data_path", default=None, help="path to training dataset")
    parser.add_option("-m","--mask", action="store", type="str", dest="mask_path", default=None, help="path to mask")
    parser.add_option("-o","--out",  action="store", type="str", dest="out_path",  default=None, help="path to output PKL file")
    parser.add_option("-c","--caps", action="store", type="str", dest="caps_path", default=None, help="path to caps template")
    parser.add_option("--discard",   action="store", type="int", dest="nvols_discard",   default=100,  help="number of volumes")
    return parser.parse_args(options)

def main():
    # 1) Read Input Parameters
    log.info('1) Reading Program Inputs...')
    opts, args = processProgramOptions(sys.argv)
    log.debug('User Options: %s' % str(opts))

    # 2) Ensure Inputs Correctness
    if opts.mask_path is None:
        log.error('Mask is missing. Please provide one.')
        sys.exit(-1)
    if opts.data_path is None:
        log.error('Training data is missing. Please provide one.')
        sys.exit(-1)
    if opts.out_path is None:
        log.error('Output path is missing. Please provide one.')
        sys.exit(-1)
    if opts.caps_path is None:
        log.error('Path to CAPs templates is missing. Please provide one.')
        sys.exit(-1)
    
    # 3) Initialize Progrma Object
    program = Program(opts)

    # 4) Load Datasets into memory
    program.load_datasets()

    # 5) Generate Training labels via Linear Regression + Z-scoring
    program.generate_training_labels()

    # 5) Train the SVRs
    program.train_svrs()
    
    
    
if __name__ == '__main__':
   sys.exit(main())