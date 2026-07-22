import numpy as np
import nibabel as nib
import pytest

from rtcog.matching.offline.nmi import OfflineNMI, load_template_labels


def test_load_template_labels_defaults_to_generic_template_labels():
    labels = load_template_labels(None, 3)

    assert labels == ["T01", "T02", "T03"]


def test_load_template_labels_rejects_wrong_count(tmp_path):
    labels_path = tmp_path / "labels.txt"
    labels_path.write_text("one,two")

    with pytest.raises(ValueError, match="2 labels for 3 templates"):
        load_template_labels(str(labels_path), 3)


def test_offline_nmi_builds_template_data_from_4d_nifti(tmp_path):
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

    offline_nmi = OfflineNMI(
        templates_path=str(templates_path),
        mask_path=str(mask_path),
        template_labels_path=str(labels_path),
    )
    template_data = offline_nmi.build_template_data()

    assert template_data["labels"].tolist() == ["template_one", "template_two"]
    assert template_data["templates"].shape == (2, 8)
    assert template_data["template_bins"].shape == (2, 8)
    assert template_data["n_bins"] == 2

    stats_tables = offline_nmi.build_template_stats_tables()
    assert set(stats_tables) == {
        "nmi_template_pairwise_stats",
    }

    pairwise = stats_tables["nmi_template_pairwise_stats"].iloc[0]
    assert pairwise["template_a"] == "template_one"
    assert pairwise["template_b"] == "template_two"
    assert pairwise["overlap_voxels"] == 7
    assert pairwise["spatial_pearson_r"] == -1
    assert set(pairwise.index) == {
        "template_a",
        "template_b",
        "overlap_voxels",
        "spatial_pearson_r",
    }


def test_offline_nmi_builds_template_data_from_3d_nifti(tmp_path):
    mask_path = tmp_path / "mask.nii"
    templates_path = tmp_path / "template.nii"

    mask = np.ones((2, 2, 2), dtype=np.float32)
    template = np.arange(8, dtype=np.float32).reshape((2, 2, 2), order="F")

    nib.Nifti1Image(mask, np.eye(4)).to_filename(mask_path)
    nib.Nifti1Image(template, np.eye(4)).to_filename(templates_path)

    offline_nmi = OfflineNMI(
        templates_path=str(templates_path),
        mask_path=str(mask_path),
    )
    template_data = offline_nmi.build_template_data()

    assert template_data["labels"].tolist() == ["T01"]
    assert template_data["templates"].shape == (1, 8)
    assert template_data["template_bins"].shape == (1, 8)


def test_offline_nmi_returns_signed_scores_after_discard(tmp_path):
    mask_path = tmp_path / "mask.nii"
    templates_path = tmp_path / "templates.nii"
    data_path = tmp_path / "data.nii"
    labels_path = tmp_path / "labels.txt"

    mask = np.ones((2, 2, 2), dtype=np.float32)
    positive = np.arange(8, dtype=np.float32).reshape((2, 2, 2), order="F")
    negative = np.arange(7, -1, -1, dtype=np.float32).reshape((2, 2, 2), order="F")
    templates = np.stack([positive, negative], axis=-1)

    data = np.zeros((2, 2, 2, 3), dtype=np.float32)
    data[..., 0] = positive
    data[..., 1] = positive
    data[..., 2] = negative

    nib.Nifti1Image(mask, np.eye(4)).to_filename(mask_path)
    nib.Nifti1Image(templates, np.eye(4)).to_filename(templates_path)
    nib.Nifti1Image(data, np.eye(4)).to_filename(data_path)
    labels_path.write_text("positive,negative")

    offline_nmi = OfflineNMI(
        templates_path=str(templates_path),
        mask_path=str(mask_path),
        template_labels_path=str(labels_path),
        data_path=str(data_path),
        discard=1,
        out_dir=str(tmp_path),
        prefix="demo",
    )
    score_data = offline_nmi.score_data()

    assert score_data["scores"].shape == (2, 3)
    np.testing.assert_array_equal(score_data["scores"][:, 0], [0, 0])
    np.testing.assert_allclose(score_data["scores"][:, 1], [1, -1])
    np.testing.assert_allclose(score_data["scores"][:, 2], [-1, 1])

    score_paths = offline_nmi.save_score_data()
    np.testing.assert_array_equal(np.load(score_paths["scores"]), score_data["scores"])
    saved_traces = np.load(score_paths["traces"])
    np.testing.assert_array_equal(saved_traces["positive"], score_data["scores"][0])


def test_offline_nmi_saves_template_data_round_trip_npz(tmp_path):
    template_data = {
        "labels": np.array(["a"]),
        "templates": np.array([[0, 1, 2]], dtype=np.float32),
        "template_bins": np.array([[1, 2, 2]], dtype=np.int16),
        "n_bins": np.array(2),
    }

    offline_nmi = OfflineNMI.from_template_data(
        template_data,
        out_dir=str(tmp_path),
        prefix="demo",
    )
    out_path = offline_nmi.save_template_data()

    saved = np.load(out_path, allow_pickle=True)
    assert saved["labels"].tolist() == ["a"]
    np.testing.assert_array_equal(saved["templates"], template_data["templates"])
    np.testing.assert_array_equal(saved["template_bins"], template_data["template_bins"])


def test_offline_nmi_save_template_data_requires_existing_dir(tmp_path):
    missing_dir = tmp_path / "missing"
    offline_nmi = OfflineNMI.from_template_data(
        {},
        out_dir=str(missing_dir),
        prefix="demo",
    )

    with pytest.raises(FileNotFoundError, match="Out directory"):
        offline_nmi.save_template_data()
