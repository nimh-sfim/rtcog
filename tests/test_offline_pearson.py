from unittest.mock import patch

import nibabel as nib
import numpy as np

from rtcog.matching.matcher import PearsonMatcher
from rtcog.matching.matching_opts import MatchingOpts
from rtcog.matching.offline.pearson import OfflinePearson


def _write_spatial_inputs(tmp_path):
    mask_path = tmp_path / "mask.nii"
    templates_path = tmp_path / "templates.nii"
    data_path = tmp_path / "data.nii"
    labels_path = tmp_path / "labels.txt"

    mask = np.ones((2, 2, 2), dtype=np.float32)
    positive = np.arange(8, dtype=np.float32).reshape((2, 2, 2), order="F")
    negative = np.arange(7, -1, -1, dtype=np.float32).reshape(
        (2, 2, 2), order="F"
    )
    templates = np.stack([positive, negative], axis=-1)
    data = np.stack([positive, positive, negative], axis=-1)

    nib.Nifti1Image(mask, np.eye(4)).to_filename(mask_path)
    nib.Nifti1Image(templates, np.eye(4)).to_filename(templates_path)
    nib.Nifti1Image(data, np.eye(4)).to_filename(data_path)
    labels_path.write_text("positive,negative")
    return mask_path, templates_path, data_path, labels_path


def test_offline_pearson_prepares_its_own_runtime_artifact(tmp_path):
    mask_path, templates_path, _, labels_path = _write_spatial_inputs(tmp_path)
    workflow = OfflinePearson(
        templates_path=str(templates_path),
        mask_path=str(mask_path),
        template_labels_path=str(labels_path),
        out_dir=str(tmp_path),
        prefix="demo",
    )

    outputs = workflow.run()

    assert set(outputs) == {"templates"}
    assert outputs["templates"].endswith("demo.pearson_templates.npz")
    artifact = np.load(outputs["templates"])
    assert set(artifact.files) == {"labels", "templates"}
    assert artifact["labels"].tolist() == ["positive", "negative"]
    assert artifact["templates"].shape == (2, 8)


def test_offline_pearson_scores_data_in_the_preparation_run(tmp_path):
    mask_path, templates_path, data_path, labels_path = _write_spatial_inputs(
        tmp_path
    )
    workflow = OfflinePearson(
        templates_path=str(templates_path),
        mask_path=str(mask_path),
        template_labels_path=str(labels_path),
        data_path=str(data_path),
        discard=1,
        out_dir=str(tmp_path),
        prefix="demo",
    )

    outputs = workflow.run()

    assert set(outputs) == {"templates", "scores", "traces", "plot"}
    scores = np.load(outputs["scores"])
    np.testing.assert_array_equal(scores[:, 0], [0, 0])
    np.testing.assert_allclose(scores[:, 1], [1, -1])
    np.testing.assert_allclose(scores[:, 2], [-1, 1])
    traces = np.load(outputs["traces"])
    np.testing.assert_array_equal(traces["positive"], scores[0])


@patch.object(PearsonMatcher, "setup_shared_memory")
def test_pearson_matcher_loads_offline_pearson_artifact(
    mock_setup_shared_memory,
    tmp_path,
    make_sync_mock,
):
    mask_path, templates_path, _, labels_path = _write_spatial_inputs(tmp_path)
    workflow = OfflinePearson(
        templates_path=str(templates_path),
        mask_path=str(mask_path),
        template_labels_path=str(labels_path),
        out_dir=str(tmp_path),
        prefix="demo",
    )
    artifact_path = workflow.run()["templates"]
    sync = make_sync_mock()

    matcher = PearsonMatcher(
        MatchingOpts(match_method="pearson", match_start=0, vols_noaction=4),
        Nt=3,
        sync=sync,
        match_path=artifact_path,
    )

    assert matcher.template_labels == ["positive", "negative"]
    assert matcher.Nvoxels == 8
    mock_setup_shared_memory.assert_called_once()
    sync.shm_ready.set.assert_called_once()
