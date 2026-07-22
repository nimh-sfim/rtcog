import numpy as np
import pandas as pd
import pytest

from rtcog.matching.offline import svr
from rtcog.matching.offline.svr_training import SVRtrainer


def _touch_inputs(tmp_path):
    paths = {
        "data": tmp_path / "data.nii",
        "templates": tmp_path / "templates.nii",
        "labels": tmp_path / "labels.txt",
        "mask": tmp_path / "mask.nii",
    }
    for path in paths.values():
        path.write_text("fake")
    return paths


def test_process_program_options_parses_fake_inputs(tmp_path):
    paths = _touch_inputs(tmp_path)

    opts = svr.processProgramOptions(
        None,
        [
            "-d",
            str(paths["data"]),
            "-t",
            str(paths["templates"]),
            "-l",
            str(paths["labels"]),
            "-m",
            str(paths["mask"]),
            "--discard",
            "2",
            "-o",
            str(tmp_path),
            "-p",
            "demo_svr",
            "--no_lasso",
        ],
    )

    assert opts.data_path == str(paths["data"])
    assert opts.templates_path == str(paths["templates"])
    assert opts.template_labels_path == str(paths["labels"])
    assert opts.mask_path == str(paths["mask"])
    assert opts.nvols_discard == 2
    assert opts.outdir == str(tmp_path)
    assert opts.prefix == "demo_svr"
    assert opts.no_lasso is True


@pytest.mark.parametrize(
    ("extra_args", "expected_do_lasso"),
    [
        ([], True),
        (["--no_lasso"], False),
    ],
)
def test_main_runs_expected_svr_pipeline_without_real_outputs(
    monkeypatch,
    tmp_path,
    extra_args,
    expected_do_lasso,
):
    paths = _touch_inputs(tmp_path)
    trainer_instances = []

    class FakeSVRTrainer:
        def __init__(self, opts):
            self.opts = opts
            self.do_lasso = not opts.no_lasso
            self.calls = []
            trainer_instances.append(self)

        def load_datasets(self):
            self.calls.append("load_datasets")

        def generate_training_labels(self):
            self.calls.append("generate_training_labels")

        def train_svrs_mp(self):
            self.calls.append("train_svrs_mp")

        def save_results(self):
            self.calls.append("save_results")

    argv = [
        "svr.py",
        "-d",
        str(paths["data"]),
        "-t",
        str(paths["templates"]),
        "-l",
        str(paths["labels"]),
        "-m",
        str(paths["mask"]),
        "--discard",
        "1",
        "-o",
        str(tmp_path),
        "-p",
        "demo_svr",
        *extra_args,
    ]

    monkeypatch.setattr(svr.sys, "argv", argv)
    monkeypatch.setattr(svr, "SVRtrainer", FakeSVRTrainer)

    assert svr.main() == 1

    trainer = trainer_instances[0]
    assert trainer.do_lasso is expected_do_lasso
    assert trainer.calls == [
        "load_datasets",
        "generate_training_labels",
        "train_svrs_mp",
        "save_results",
    ]


def test_generate_training_labels_maps_fake_matrices_to_expected_outputs():
    trainer = SVRtrainer.__new__(SVRtrainer)
    trainer.nvols_discard = 1
    trainer.data_nt = 5
    trainer.do_lasso = False
    trainer.template_labels = ["signal_a", "signal_b"]
    trainer.templates_masked = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ],
        dtype=float,
    )

    expected_raw_labels = np.array(
        [
            [0, 0],
            [1, 5],
            [2, 4],
            [3, 3],
            [4, 2],
        ],
        dtype=float,
    )
    intercept = 10
    trainer.data_masked = intercept + trainer.templates_masked @ expected_raw_labels.T

    trainer.generate_training_labels()

    expected_z_labels = np.array(
        [
            [-1.47709789, -1.47709789],
            [-0.86164044, 1.60018938],
            [-0.24618298, 0.98473193],
            [0.36927447, 0.36927447],
            [0.98473193, -0.24618298],
        ]
    )

    np.testing.assert_array_equal(trainer.vols4training, [1, 2, 3, 4])
    np.testing.assert_allclose(trainer.lm_R2, [0, 1, 1, 1, 1])
    pd.testing.assert_frame_equal(
        trainer.LR_labels_preZscore,
        pd.DataFrame(expected_raw_labels, columns=trainer.template_labels),
    )
    np.testing.assert_allclose(trainer.lm_res_z.to_numpy(), expected_z_labels)
