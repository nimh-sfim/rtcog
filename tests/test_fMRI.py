import numpy as np

from rtcog.utils.fMRI import mask_fMRI_img, unmask_fMRI_img


def test_synthetic_fmri_data_has_required_signal_variation(synthetic_fmri_data):
    data = synthetic_fmri_data
    in_mask = data.data[data.mask]

    assert data.data.ndim == 4
    assert data.n_volumes > 1
    assert 0 < data.n_voxels < np.prod(data.mask.shape)
    assert np.ptp(in_mask[:, 0]) > 0
    assert np.any(np.ptp(in_mask, axis=1) > 0)


def test_mask_fMRI_img(synthetic_fmri_data):
    data = synthetic_fmri_data
    masked_data = mask_fMRI_img(data.img, data.mask_img)
    flat_indices = np.flatnonzero(data.mask.reshape(-1, order="F"))
    coordinates = np.unravel_index(flat_indices, data.mask.shape, order="F")
    expected = data.data[coordinates]

    assert masked_data.shape == (data.n_voxels, data.n_volumes)
    np.testing.assert_array_equal(masked_data, expected)


def test_unmask_fMRI_img_round_trips_masked_voxels(synthetic_fmri_data):
    data = synthetic_fmri_data
    masked_data = mask_fMRI_img(data.img, data.mask_img)
    unmasked_data = unmask_fMRI_img(masked_data, data.mask_img)
    expected = np.where(data.mask[..., np.newaxis], data.data, 0)

    assert unmasked_data.shape == data.img.shape
    np.testing.assert_array_equal(unmasked_data, expected)


if __name__ == "__main__":
    pytest.main()
