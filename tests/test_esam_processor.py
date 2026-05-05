import os.path as osp
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from rtcog.processor.esam_processor import ESAMProcessor
from rtcog.preproc.step_types import StepType


def make_processor_for_hit_maps(tmp_path, *, windowing=True):
    processor = ESAMProcessor.__new__(ESAMProcessor)
    processor.out_dir = str(tmp_path)
    processor.out_prefix = "test"
    processor.mask_img = MagicMock()
    processor.mask_Nv = 10
    processor.hit_opts = SimpleNamespace(nconsec_vols=3)
    processor.log = MagicMock()

    processor.matcher = MagicMock()
    processor.matcher.template_labels = ["dmn", "vis"]

    processor.hits = np.zeros((2, 6), dtype=bool)
    processor.hits[0, 5] = True

    processor.pipe = MagicMock()
    processor.pipe.Data_processed = np.arange(60).reshape(10, 6)
    processor.pipe.step_opts = []
    if windowing:
        processor.pipe.step_opts.append({
            "name": StepType.WINDOWING.value,
            "enabled": True,
            "win_length": 4,
        })

    return processor


@patch("rtcog.processor.esam_processor.unmask_fMRI_img")
def test_write_hit_maps_includes_windowed_contributing_volumes(unmask_mock, tmp_path):
    processor = make_processor_for_hit_maps(tmp_path, windowing=True)

    processor.write_hit_maps()

    assert unmask_mock.call_count == 1
    img_data, mask_img_arg, out_path = unmask_mock.call_args[0]

    expected_vols = np.array([5, 4, 3, 2, 1, 0])
    expected_data = processor.pipe.Data_processed[:, expected_vols].mean(axis=1)

    np.testing.assert_array_equal(img_data, expected_data)
    assert mask_img_arg is processor.mask_img
    assert out_path == osp.join(str(tmp_path), "test.Hit_dmn_01.nii")


@patch("rtcog.processor.esam_processor.unmask_fMRI_img")
def test_write_hit_maps_uses_nconsec_without_windowing(unmask_mock, tmp_path):
    processor = make_processor_for_hit_maps(tmp_path, windowing=False)

    processor.write_hit_maps()

    img_data = unmask_mock.call_args[0][0]
    expected_vols = np.array([5, 4, 3])
    expected_data = processor.pipe.Data_processed[:, expected_vols].mean(axis=1)

    np.testing.assert_array_equal(img_data, expected_data)
