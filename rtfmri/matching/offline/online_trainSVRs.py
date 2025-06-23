import logging
import os.path as osp
import sys
from scipy.stats import zscore
import argparse

log     = logging.getLogger("trainSVRs")
log_fmt = logging.Formatter('[%(levelname)s - Main]: %(message)s')
log_ch  = logging.StreamHandler()
log_ch.setFormatter(log_fmt)
log.setLevel(logging.INFO)
log.addHandler(log_ch)


sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))

from tqdm import tqdm
tqdm().pandas()

from rtfmri.matching.offline.training import SVRtrainer
from rtfmri.utils.core import file_exists
# -------------------------------------------------------------------------------------

def processProgramOptions (self, options=None):
    parser = argparse.ArgumentParser(description="Train SVRs for spatial template matching")
    parser_inopts = parser.add_argument_group('Input Options','Inputs to this program')
    parser_inopts.add_argument("-d","--data", action="store", type=str, dest="data_path", default=None, help="path to training dataset [Default: %(default)s]", required=True)
    parser_inopts.add_argument("-t", "--templates_path",       help="Path to templates file",     dest="templates_path", action="store", type=file_exists, default=None, required=True)
    parser_inopts.add_argument("-l", "--template_labels_path",       help="Path to text file containing comma-separated template labels in order",     dest="template_labels_path", action="store", type=file_exists, default=None, required=True)
    parser_inopts.add_argument("-m","--mask", action="store", type=str, dest="mask_path", default=None, help="path to mask [Default: %(default)s]", required=True)
    # parser_inopts.add_argument("-c","--caps", action="store", type=str, dest="caps_path", default=None, help="path to caps template [Default: ]", required=True)
    parser_inopts.add_argument("--discard",   action="store", type=int, dest="nvols_discard",   default=100,  help="number of volumes [Default: %(default)s]")
    parser_outopts = parser.add_argument_group('Output Options','Were to save results')
    parser_outopts.add_argument("-o","--outdir",  action="store", type=str, dest="outdir",  default='./', help="output directory [Default: %(default)s]")
    parser_outopts.add_argument("-p","--prefix", action="store", type=str, dest="prefix", default="svr", help="prefix for output file [Default: %(default)s]")
    parser_svropts = parser.add_argument_group('Training Options','Different Training Options')
    parser_svropts.add_argument("--no_lasso",          action="store_true", default=False, dest="no_lasso", help="Generate Labels with Linear Regression (No Lasso) [Default: %(default)s]")
    parser_svropts.add_argument("--lasso_alpha",       action="store",      type=float, default=0.75,  dest="lasso_alpha", help="Regularization constant for Lasso Step (Label Generation) [Default: %(default)s]")
    parser_svropts.add_argument("--lasso_no_pos_only", action="store",      default=False, dest="lasso_no_pos_only", help="Allow positives and negative fit values in Lasso Step (Label Generation) [Default: %(default)s]")
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
    if opts.templates_path is None:
        log.error('Path to templates is missing. Please provide one.')
        sys.exit(-1)
    
    # 3) Initialize Program Object
    log.info('2) Initializing Program Object...')
    svr_trainer = SVRtrainer(opts)

    # 4) Load Datasets into memory
    log.info('3) Load all data into memory...')
    svr_trainer.load_datasets()

    # 5) Generate Training labels via Linear Regression + Z-scoring
    if svr_trainer.do_lasso:
        log.info('4) Generate traning labels (LASSO)...')
        svr_trainer.generate_training_labels_lasso()
    else:
        log.info('4) Generate traning labels (Linear Regression)...')
        svr_trainer.generate_training_labels()

    # 5) Train the SVRs
    log.info('5) Train SVRs...')
    svr_trainer.train_svrs_mp()

    # 6) Save results to disk
    log.info('6) Save results to disk...')
    svr_trainer.save_results()
    
    return 1
    
if __name__ == '__main__':
   sys.exit(main())