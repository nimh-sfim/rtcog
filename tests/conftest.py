import sys
import os.path as osp
import pytest
import numpy as np
import pandas as pd
import nibabel as nib

from rtcog.paths import DATA_DIR

def pytest_addoption(parser):
    parser.addoption(
        "--snapshot", action="store_true", default=False, help="run snapshot test"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--snapshot"):
        return
    skip_snapshot = pytest.mark.skip(reason="need --snapshot option to run")
    for item in items:
        if "snapshot" in item.keywords:
            item.add_marker(skip_snapshot)
            
class SampleData:
    def __init__(self):
        self.orig_img = nib.load(osp.join(DATA_DIR, 'test_epi_data100+orig.BRIK.gz'))
        self.mask_img = nib.load(osp.join(DATA_DIR, 'GMribbon_R4Feed.nii'))

        self.orig_data, self.mask_data = self.orig_img.get_fdata(), self.mask_img.get_fdata()

        self.this_t_data = np.load(osp.join(DATA_DIR, 'this_t_data.npy'))

        self.Nv = len(self.this_t_data)
        self.t = self.orig_data.shape[-1]
        self.n = self.t - 10
    

@pytest.fixture(scope="session")
def sample_data():
    return SampleData()