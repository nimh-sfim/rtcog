import sys
import os.path as osp
import pytest
import numpy as np
import pandas as pd
import nibabel as nib

sys.path.append('..')
from config import DATA_DIR


class SampleData:
    def __init__(self):
        self.orig_img = nib.load(osp.join(DATA_DIR, 'test_epi_data100+orig.BRIK.gz'))
        self.mask_img = nib.load(osp.join(DATA_DIR, 'GMribbon_R4Feed.nii'))

        self.orig_data, self.mask_data = self.orig_img.get_fdata(), self.mask_img.get_fdata()

        self.this_t_data = pd.read_pickle(osp.join(DATA_DIR, 'this_t_data.pkl'))

        self.Nv = len(self.this_t_data)
        self.t = self.orig_data.shape[-1]
        self.n = self.t - 10
    
    def get_imgs(self):
        return self.orig_img, self.mask_img
    
    def get_arrs(self):
        return self.orig_data, self.mask_data
    
    def get_this_t_data(self):
        return self.this_t_data
    

@pytest.fixture(scope="class")
def sample_data():
    return SampleData()