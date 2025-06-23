import pytest
import numpy as np

from rtfmri.utils.fMRI import mask_fMRI_img, unmask_fMRI_img

def test_mask_fMRI_img(sample_data):
    masked_data = mask_fMRI_img(sample_data.orig_img,sample_data. mask_img)
    expected_shape = (np.sum(sample_data.mask_img.get_fdata() == 1), sample_data.orig_img.shape[3])

    assert masked_data.shape == expected_shape
    assert masked_data.shape[0] == sample_data.Nv
    assert masked_data.shape[1] == sample_data.t


def test_unmask_fMRI_img(sample_data):
    masked_data = mask_fMRI_img(sample_data.orig_img, sample_data.mask_img)
    unmasked_data = unmask_fMRI_img(masked_data, sample_data.mask_img)

    assert unmasked_data.shape == sample_data.orig_img.shape


if __name__ == "__main__":
    pytest.main()