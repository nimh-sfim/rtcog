import sys
import argparse
import os.path as osp
import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas
import panel as pn

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))
from rtcap_lib.fMRI import load_fMRI_file, mask_fMRI_img
from bin.rtcaps_matcher import file_exists

import logging
log     = logging.getLogger("offline_mask")
log_fmt = logging.Formatter('[%(levelname)s - offline_mask]: %(message)s')
log_ch  = logging.StreamHandler()
log.setLevel(logging.INFO)
log_ch.setFormatter(log_fmt)
log.addHandler(log_ch)

class OfflineMask:
    def __init__(self, opts):
        if not osp.isdir(opts.out_dir):
            log.error('Out directory does not exist')
            sys.exit(-1)

        self.templates_path = opts.templates_path
        self.data_path = opts.data_path
        self.mask_path = opts.mask_path
        
        self.template_labels_path = opts.template_labels_path

        self.out_dir = opts.out_dir
        self.out_prefix = opts.prefix

        self.template_thr = opts.template_thr
        
    def load_datasets(self):
        try:
            with open (self.template_labels_path, 'r') as f:
                lines = f.read()
                self.template_labels = [label.strip() for label in lines.strip().split(',')]
        except Exception as e:
            log.error(e)
            sys.exit(-1)

        log.info(f"Processing templates: {', '.join(self.template_labels)}")
        self.data_img = load_fMRI_file(self.data_path)
        self.mask_img = load_fMRI_file(self.mask_path)
        self.templates_img = load_fMRI_file(self.templates_path)
        
        masked_template_array = mask_fMRI_img(self.templates_img, self.mask_img)
        self.templates_masked = [masked_template_array[:, i] for i in range(masked_template_array.shape[1])]
        self.data_masked = mask_fMRI_img(self.data_img, self.mask_img)
        log.debug(f'Masked data dimensions: {self.data_masked.shape}')

    def _threshold(self, template):
        """Select voxels for a template above desired threshold and apply that mask to the data."""
        composite_mask_vect = (template > int(self.template_thr)).astype(bool)
        composite_mask_Nv = composite_mask_vect.sum()

        thresholded_template = template[composite_mask_vect]

        data_vect = self.data_masked
        log.debug(f"template shape: {template.shape}")
        log.debug(f"data_vect shape: {self.data_masked.shape}")
        log.debug(f"composite_mask_vect shape: {composite_mask_vect.shape}")
        thresholded_data = data_vect[composite_mask_vect, :]

        return thresholded_template, thresholded_data, composite_mask_Nv

    def get_masked_traces(self):
        """Compute and save masked activation traces and thresholded template data."""
        self.act_traces = {}

        masked_templates = {}
        mask_vectors = {}
        voxel_counts = {}

        for label, template in zip(self.template_labels, self.templates_masked):
            thr_template, thr_data, Nvoxels_in_mask = self._threshold(template)
            act_trace = np.dot(thr_template, thr_data) / Nvoxels_in_mask
            
            self.act_traces[label] = act_trace
            masked_templates[label] = thr_template
            mask_vectors[label] = (template > int(self.template_thr)).astype(bool).flatten(order='F')
            voxel_counts[label] = Nvoxels_in_mask

        # Save activation traces
        trace_out = osp.join(self.out_dir, f'{self.out_prefix}_self.act_traces.npz')
        np.savez(trace_out, **self.act_traces)

        # Save thresholded template info for online use
        template_out = osp.join(self.out_dir, f'{self.out_prefix}_template_data.npz')
        np.savez(
            template_out,
            labels=np.array(self.template_labels),
            masked_templates=masked_templates,
            masks=mask_vectors,
            voxel_counts=voxel_counts
        )

        log.info(f'Saved traces to: {trace_out}')
        log.info(f'Saved thresholded template data to: {template_out}')
    
    def plot_traces(self):
        df = pd.DataFrame.from_dict(self.act_traces, orient='index').T
        plot = df.hvplot(width=1000)
        out = osp.join(self.out_dir, f'{self.out_prefix}_traces')
        # out_html = 
        pn.Column(plot).save(out + '.html')

def process_options():
    parser = argparse.ArgumentParser(description="Train SVRs for spatial template matching")
    parser_inopts = parser.add_argument_group('Input Options','Inputs to this program')
    parser_inopts.add_argument("-d","--data", action="store", type=file_exists, dest="data_path", default=None, help="path to training dataset", required=True)
    parser_inopts.add_argument("-t", "--templates_path", help="Path to templates file", dest="templates_path", action="store", type=file_exists, default=None, required=True)
    parser_inopts.add_argument("-l", "--template_labels_path", help="Path to text file containing comma-separated template labels in order", dest="template_labels_path", action="store", type=file_exists, default=None, required=True)
    parser_inopts.add_argument("-m","--mask", action="store", type=file_exists, dest="mask_path", default=None, help="path to mask", required=True)
    parser_inopts.add_argument("--discard", action="store", type=int, dest="nvols_discard", default=100,  help="number of volumes to discard")
    parser_inopts.add_argument("--thr", action="store", type=float, dest="template_thr", default=10,  help="threshold to use for the templates [Default: %(default)s]")
    parser_inopts.add_argument("--debug", action="store_true", dest="debug", default=False,  help="Enable debugging [Default: False]")
    parser_outopts = parser.add_argument_group('Output Options','Were to save results')
    parser_outopts.add_argument("-o","--out_dir",  action="store", type=str, dest="out_dir", default='./', help="output directory [Default: %(default)s]")
    parser_outopts.add_argument("-p","--prefix", action="store", type=str, dest="prefix", default="mask_method", help="prefix for output file [Default: %(default)s]")
    return parser.parse_args()  

if __name__ == "__main__":
    opts = process_options()
    if opts.debug:
        log.setLevel(logging.DEBUG)
    offline_mask = OfflineMask(opts)
    offline_mask.load_datasets()
    offline_mask.get_masked_traces()
    offline_mask.plot_traces()
 
# npz = np.load("template_data.npz", allow_pickle=True)
# labels = npz["labels"]
# masked_templates = npz["masked_templates"].item()
# masks = npz["masks"].item()
# voxel_counts = npz["voxel_counts"].item()
