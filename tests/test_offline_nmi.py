import numpy as np
import nibabel as nib
import pytest

from rtcog.matching.offline.nmi import (
    build_nmi_template_data,
    load_template_labels,
    save_nmi_template_data,
)


def test_load_template_labels_defaults_to_generic_template_labels():
    labels = load_template_labels(None, 3)

    assert labels == ["T01", "T02", "T03"]


def test_load_template_labels_rejects_wrong_count(tmp_path):
    labels_path = tmp_path / "labels.txt"
    labels_path.write_text("one,two")

    with pytest.raises(ValueError, match="2 labels for 3 templates"):
        load_template_labels(str(labels_path), 3)


def test_build_nmi_template_data_from_4d_nifti(tmp_path):
    mask_path = tmp_path / "mask.nii"
    templates_path = tmp_path / "templates.nii"
    labels_path = tmp_path / "labels.txt"

    mask = np.ones((2, 2, 2), dtype=np.float32)
    templates = np.zeros((2, 2, 2, 2), dtype=np.float32)
    templates[..., 0] = np.arange(8, dtype=np.float32).reshape((2, 2, 2), order="F")
    templates[..., 1] = np.arange(8, 0, -1, dtype=np.float32).reshape((2, 2, 2), order="F")

    nib.Nifti1Image(mask, np.eye(4)).to_filename(mask_path)
    nib.Nifti1Image(templates, np.eye(4)).to_filename(templates_path)
    labels_path.write_text("template_one,template_two")

    template_data = build_nmi_template_data(
        str(templates_path),
        str(mask_path),
        str(labels_path),
    )

    assert template_data["labels"].tolist() == ["template_one", "template_two"]
    assert template_data["templates"].shape == (2, 8)
    assert template_data["template_bins"].shape == (2, 8)
    assert template_data["n_bins"] == 2


def test_build_nmi_template_data_from_3d_nifti(tmp_path):
    mask_path = tmp_path / "mask.nii"
    templates_path = tmp_path / "template.nii"

    mask = np.ones((2, 2, 2), dtype=np.float32)
    template = np.arange(8, dtype=np.float32).reshape((2, 2, 2), order="F")

    nib.Nifti1Image(mask, np.eye(4)).to_filename(mask_path)
    nib.Nifti1Image(template, np.eye(4)).to_filename(templates_path)

    template_data = build_nmi_template_data(str(templates_path), str(mask_path))

    assert template_data["labels"].tolist() == ["T01"]
    assert template_data["templates"].shape == (1, 8)
    assert template_data["template_bins"].shape == (1, 8)


def test_save_nmi_template_data_round_trips_npz(tmp_path):
    template_data = {
        "labels": np.array(["a"]),
        "templates": np.array([[0, 1, 2]], dtype=np.float32),
        "template_bins": np.array([[1, 2, 2]], dtype=np.int16),
        "n_bins": np.array(2),
    }

    out_path = save_nmi_template_data(template_data, str(tmp_path), "demo")

    saved = np.load(out_path, allow_pickle=True)
    assert saved["labels"].tolist() == ["a"]
    np.testing.assert_array_equal(saved["templates"], template_data["templates"])
    np.testing.assert_array_equal(saved["template_bins"], template_data["template_bins"])


def test_save_nmi_template_data_requires_existing_dir(tmp_path):
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Out directory"):
        save_nmi_template_data({}, str(missing_dir), "demo")
