import sys
import argparse
import os.path as osp
import re
import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas
import panel as pn
import matplotlib.pyplot as plt

from rtcog.utils.fMRI import load_fMRI_file, mask_fMRI_img
from rtcog.utils.core import file_exists

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
        
        if not opts.no_calc and opts.data_path is None:
            log.error("--data is required unless --no_calc is used")
            sys.exit(-1)

        self.no_calc = opts.no_calc

        self.templates_path = opts.templates_path
        self.data_path = opts.data_path
        self.mask_path = opts.mask_path
        self.template_labels_path = opts.template_labels_path

        self.template_thr = opts.template_thr
        self.template_type = opts.template_type
        
        self.nvols_discard = opts.nvols_discard

        self.out_path = osp.join(opts.out_dir, opts.prefix)
        self.save_txt = opts.save_txt
        
    def load_datasets(self):
        try:
            with open (self.template_labels_path, 'r') as f:
                lines = f.read()
                self.template_labels = re.split(r"[,\s]+", lines.strip())
        except Exception as e:
            log.error(e)
            sys.exit(-1)

        log.info(f"Processing templates: {', '.join(self.template_labels)}")
        mask_img = load_fMRI_file(self.mask_path)
        self.templates_img = load_fMRI_file(self.templates_path)
        
        if mask_img.shape != self.templates_img.shape[:3]:
            raise ValueError("Template volume and mask volume have incompatible shapes")

        masked_template_array = mask_fMRI_img(self.templates_img, mask_img)

        self.templates_masked = [masked_template_array[:, i] for i in range(masked_template_array.shape[1])]

        if not self.no_calc:
            self.training_img = load_fMRI_file(self.data_path)
            full_training_masked = mask_fMRI_img(self.training_img, mask_img)
            self.training_masked = full_training_masked[:, self.nvols_discard:]
            log.debug(f'Masked data dimensions: {self.training_masked.shape}')

    def _threshold(self, label, template):
        """Select voxels for a template above desired threshold and apply that mask to the data."""
        if self.template_type == 'normal':
            composite_mask_vect = (template > int(self.template_thr)).astype(bool)
        elif self.template_type == 'binary':
            composite_mask_vect = template.astype(bool)
        
        self.mask_vectors[label] = composite_mask_vect
            
        composite_mask_Nv = composite_mask_vect.sum()

        thresholded_template = template[composite_mask_vect]

        log.debug(f"template shape: {template.shape}")
        log.debug(f"composite_mask_vect shape: {composite_mask_vect.shape}")
        
        thresholded_training = None
        if not self.no_calc:
            log.debug(f"training_masked.shape: {self.training_masked.shape}")
            thresholded_training = self.training_masked[composite_mask_vect, :]

        return thresholded_template, thresholded_training, composite_mask_Nv

    def get_masked_data(self):
        """Compute and save masked activation traces and thresholded template data."""
        self.act_traces = {}

        masked_templates = {}
        voxel_counts = {}
        self.mask_vectors = {}
        
        for label, template in zip(self.template_labels, self.templates_masked):
            thr_template, thr_training, Nvoxels_in_mask = self._threshold(label, template)
            if Nvoxels_in_mask == 0:
                log.error(f"Template '{label}' has zero voxels after thresholding.")
                continue
            masked_templates[label] = thr_template
            voxel_counts[label] = Nvoxels_in_mask

            if not self.no_calc:
                act_trace = np.dot(thr_template, thr_training) / Nvoxels_in_mask
                
                full_timepoints = self.training_masked.shape[1] + self.nvols_discard
                final = np.zeros(full_timepoints)
                final[self.nvols_discard:] = act_trace

                self.act_traces[label] = final

        # Save thresholded template info for online use
        template_out = self.out_path + '.template_data.npz'
        np.savez(
            template_out,
            labels=np.array(self.template_labels),
            masked_templates=masked_templates,
            masks=self.mask_vectors,
            voxel_counts=voxel_counts
        )
        log.info(f'Saved thresholded template data to: {template_out}')

        # Save activation traces
        if not self.no_calc:
            trace_out = self.out_path + '.act_traces.npz'
            np.savez(trace_out, **self.act_traces)

            if self.save_txt: # Save one txt file per trace if desired
                for template, arr in self.act_traces.items():
                    with open(f'{template}.act_trace.txt', 'w') as f:
                        np.savetxt(f, arr, delimiter=',')

            log.info(f'Saved traces to: {trace_out}')
    
    def save_figures(self):
        if self.no_calc:
            return

        df = pd.DataFrame.from_dict(self.act_traces, orient='index').T

        # Static figure
        fig = plt.figure(figsize=(20,5))
        plt.plot(df)
        plt.xlabel('Time [TRs]')
        plt.ylabel('Weighted average')
        plt.legend(self.template_labels)
        png_out = self.out_path + '.traces.png'
        plt.savefig(png_out, dpi=200)
        plt.close(fig)
        log.info(f'Saved static figure to: {png_out}')
        # Note: for some reason the file will not open through preview, but it opens just fine when loaded with Pillow,
        # so it's not corrupted.

        # Dynamic figure
        plot = df.hvplot(width=1000)
        html_out = self.out_path + '.traces.html'
        pn.Column(plot).save(html_out)
        log.info(f'Saved dynamic figure to: {html_out}')

def process_options():
    parser = argparse.ArgumentParser(description="Run mask method offline for spatial template matching")
    parser_inopts = parser.add_argument_group('Input Options','Inputs to this program')
    parser_inopts.add_argument("-d","--data", action="store", type=file_exists, dest="data_path", default=None, help="path to training dataset")
    parser_inopts.add_argument("-t", "--templates_path", help="Path to templates file", dest="templates_path", action="store", type=file_exists, default=None, required=True)
    parser_inopts.add_argument("-l", "--template_labels_path", help="Path to text file containing comma-separated template labels in order", dest="template_labels_path", action="store", type=file_exists, default=None, required=True)
    parser_inopts.add_argument("-m","--mask", action="store", type=file_exists, dest="mask_path", default=None, help="path to mask", required=True)
    parser_inopts.add_argument("--template_type", action="store", type=str, choices=['normal', 'binary'], dest="template_type", default="binary", help="the type of template being used [Default: %(default)s]")
    parser_inopts.add_argument("--discard", action="store", type=int, dest="nvols_discard", default=100,  help="number of volumes to discard")
    parser_inopts.add_argument("--thr", action="store", type=float, dest="template_thr", default=10, help="threshold to use for the templates [Default: %(default)s]")
    parser_inopts.add_argument("--no_calc", action="store_true", dest="no_calc", help="Skip calculation of activity traces and only save labels and templates")
    parser_inopts.add_argument("--debug", action="store_true", dest="debug", default=False,  help="Enable debugging [Default: False]")
    parser_outopts = parser.add_argument_group('Output Options','Where to save results')
    parser_outopts.add_argument("-o","--out_dir",  action="store", type=str, dest="out_dir", default='./', help="output directory [Default: %(default)s]")
    parser_outopts.add_argument("-p","--prefix", action="store", type=str, dest="prefix", default="mask_method", help="prefix for output file [Default: %(default)s]")
    parser_outopts.add_argument("--save_txt", action="store_true", dest="save_txt", default=False, help="save txt files for each array [Default: False]")

    return parser.parse_args()  

if __name__ == "__main__":
    opts = process_options()
    if opts.debug:
        log.setLevel(logging.DEBUG)
    offline_mask = OfflineMask(opts)
    offline_mask.load_datasets()
    offline_mask.get_masked_data()
    offline_mask.save_figures()
