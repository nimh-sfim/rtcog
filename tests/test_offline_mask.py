import numpy as np

from rtcog.matching.offline.mask import OfflineMask


def test_continuous_threshold_preserves_fractional_value():
    offline_mask = OfflineMask.__new__(OfflineMask)
    offline_mask.template_type = "continuous"
    offline_mask.template_thr = 1.5
    offline_mask.mask_vectors = {}
    offline_mask.data_masked = np.arange(8).reshape(4, 2)
    template = np.array([1.4, 1.6, 1.8, 0.0])

    thresholded_template, thresholded_data, voxel_count = offline_mask._threshold(
        "template", template
    )

    np.testing.assert_array_equal(thresholded_template, [1.6, 1.8])
    np.testing.assert_array_equal(thresholded_data, offline_mask.data_masked[1:3])
    assert voxel_count == 2
