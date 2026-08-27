import logging
import os.path as osp
import sys

log     = logging.getLogger("trainSVRs")
log_fmt = logging.Formatter('[%(levelname)s - Main]: %(message)s')
log_ch  = logging.StreamHandler()
log_ch.setFormatter(log_fmt)
log.setLevel(logging.INFO)
log.addHandler(log_ch)


sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))

from tqdm import tqdm
tqdm().pandas()

from rtcog.matching.offline.svr_training import SVRtrainer
from rtcog.matching.offline.template_utils import spatial_template_parser
# -------------------------------------------------------------------------------------

def processProgramOptions (self, options=None):
    parser, _, _ = spatial_template_parser(
        "SVR",
        "svr",
        description="Train SVRs for spatial template matching",
        data_required=True,
        labels_required=True,
        data_type=str,
        mask_type=str,
        out_dir_dest="outdir",
    )
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

    # 2) Initialize SVRTrainer Object
    log.info('2) Initializing SVRTrainer Object...')
    svr_trainer = SVRtrainer(opts)

    # 3) Load Datasets into memory
    log.info('3) Loading data into memory...')
    svr_trainer.load_datasets()

    # 4) Generate Training labels via Linear Regression + Z-scoring
    if svr_trainer.do_lasso:
        log.info('4) Generating training labels (LASSO)...')
    else:
        log.info('4) Generating training labels (Linear Regression)...')
    svr_trainer.generate_training_labels()

    # 5) Train the SVRs
    log.info('5) Training SVRs...')
    svr_trainer.train_svrs_mp()

    # 6) Save results to disk
    log.info('6) Saving results to disk...')
    svr_trainer.save_results()
    
    return 1
    
if __name__ == '__main__':
   sys.exit(main())
