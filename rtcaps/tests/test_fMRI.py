import sys
import pytest
import numpy as np

sys.path.append('../')
from rtcap_lib.fMRI import *

@pytest.fixture
def simulated_data():
    """Generates simulated fMRI data"""
    n_voxels = (64, 64, 32)
    n_t = 150
    data = np.random.rand(n_voxels[0], n_voxels[1], n_voxels[2], n_t)

    mask_data = np.zeros(n_voxels)
    mask_data[30:50, 30:50, 15:20] = 1

    affine = np.eye(4)
    data_img = nib.Nifti1Image(data, affine)
    mask_img = nib.Nifti1Image(mask_data, affine)

    return data_img, mask_img

def test_mask_fMRI_img(simulated_data):
    data_img, mask_img = simulated_data

    masked_data = mask_fMRI_img(data_img, mask_img)
    
    expected_shape = (np.sum(mask_img.get_fdata() == 1), data_img.shape[3])
    assert masked_data.shape == expected_shape, f'Expected shape {expected_shape}, but got {masked_data.shape}'

def test_unmask_fMRI_img(simulated_data):
    data_img, mask_img = simulated_data
    masked_data = mask_fMRI_img(data_img, mask_img)

    unmasked_data = unmask_fMRI_img(masked_data, mask_img)

    assert unmasked_data.shape == data_img.shape, f'Expected shape {data_img.shape}, but got {unmasked_data.shape}'

if __name__ == "__main__":
    pytest.main()