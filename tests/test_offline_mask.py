from types import SimpleNamespace

import nibabel as nib
import numpy as np

from rtcog.matching.offline.mask import OfflineMask


def _write_nifti(path, data):
    nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4)).to_filename(path)


def test_offline_mask_writes_expected_traces_and_template_data(tmp_path):
    mask_path = tmp_path / "mask.nii"
    templates_path = tmp_path / "templates.nii"
    data_path = tmp_path / "data.nii"
    labels_path = tmp_path / "labels.txt"

    mask = np.ones((3, 1, 1), dtype=np.float32)
    mask[2, 0, 0] = 0
    _write_nifti(mask_path, mask)

    templates = np.zeros((3, 1, 1, 2), dtype=np.float32)
    templates[0, 0, 0, 0] = 2
    templates[1, 0, 0, 1] = 3
    templates[2, 0, 0, :] = [99, 88]
    _write_nifti(templates_path, templates)

    data = np.zeros((3, 1, 1, 4), dtype=np.float32)
    data[0, 0, 0, :] = [10, 20, 30, 40]
    data[1, 0, 0, :] = [1, 2, 3, 4]
    _write_nifti(data_path, data)
    labels_path.write_text("left,right")

    opts = SimpleNamespace(
        templates_path=str(templates_path),
        data_path=str(data_path),
        mask_path=str(mask_path),
        template_labels_path=str(labels_path),
        template_thr=1,
        template_type="normal",
        nvols_discard=1,
        out_dir=str(tmp_path),
        prefix="demo",
        save_txt=False,
    )

    offline_mask = OfflineMask(opts)
    offline_mask.load_datasets()
    offline_mask.get_masked_traces()

    traces = np.load(tmp_path / "demo.act_traces.npz")
    np.testing.assert_array_equal(traces["left"], [0, 40, 60, 80])
    np.testing.assert_array_equal(traces["right"], [0, 6, 9, 12])

    masked_template_img = nib.load(tmp_path / "demo.masked_templates.nii.gz")
    masked_template_data = masked_template_img.get_fdata()
    expected_masked_templates = templates.copy()
    expected_masked_templates[2, 0, 0, :] = 0
    np.testing.assert_array_equal(masked_template_data, expected_masked_templates)

    template_data = np.load(tmp_path / "demo.template_data.npz", allow_pickle=True)
    assert template_data["labels"].tolist() == ["left", "right"]

    masked_templates = template_data["masked_templates"].item()
    np.testing.assert_array_equal(masked_templates["left"], [2])
    np.testing.assert_array_equal(masked_templates["right"], [3])

    masks = template_data["masks"].item()
    np.testing.assert_array_equal(masks["left"], [True, False])
    np.testing.assert_array_equal(masks["right"], [False, True])

    voxel_counts = template_data["voxel_counts"].item()
    assert voxel_counts == {"left": 1, "right": 1}

    stats_tables = offline_mask.build_stats_tables()
    assert set(stats_tables) == {
        "template_pairwise_stats",
        "trace_pairwise_stats",
    }

    template_pairwise = stats_tables["template_pairwise_stats"].iloc[0]
    assert template_pairwise["template_a"] == "left"
    assert template_pairwise["template_b"] == "right"
    assert template_pairwise["overlap_voxels"] == 0
    assert set(template_pairwise.index) == {
        "template_a",
        "template_b",
        "overlap_voxels",
        "spatial_pearson_r",
    }

    trace_pairwise = stats_tables["trace_pairwise_stats"].iloc[0]
    assert trace_pairwise["trace_a"] == "left"
    assert trace_pairwise["trace_b"] == "right"
    assert trace_pairwise["pearson_r"] == 1
    assert set(trace_pairwise.index) == {
        "trace_a",
        "trace_b",
        "pearson_r",
    }
