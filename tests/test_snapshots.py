"""Compare an rtcog run of the synthetic phantom with its saved snapshot."""

from pathlib import Path

import numpy as np
import pytest

from rtcog.paths import CONFIG_DIR, DATA_DIR, ROOT_DIR
from rtcog.utils.options import Options


EXPECTED_PATH = Path(DATA_DIR) / "synthetic_snapshot_expected.npz"
CONFIG_PATH = Path(CONFIG_DIR) / "test_default_config.yaml"
SNAPSHOT_KEYS = (
    "Data_FromAFNI",
    "Data_processed",
    "ema",
    "iglm",
    "kalman",
    "smooth",
    "snorm",
    "windowing",
)


def _actual_path() -> Path:
    config = Options.load_yaml(CONFIG_PATH)
    snapshot_dir = Path(config["snapshot_dir"])
    if not snapshot_dir.is_absolute():
        snapshot_dir = Path(ROOT_DIR) / snapshot_dir
    return snapshot_dir / "new_snapshots.npz"


@pytest.mark.snapshot
def test_synthetic_pipeline_snapshot():
    """The end-to-end synthetic run remains numerically stable."""
    actual_path = _actual_path()
    if not actual_path.exists():
        pytest.fail(
            f"Snapshot output not found at {actual_path}. "
            "Follow the snapshot-test procedure in CONTRIBUTING.rst first."
        )

    with (
        np.load(EXPECTED_PATH, allow_pickle=False) as expected,
        np.load(actual_path, allow_pickle=False) as actual,
    ):
        assert set(expected.files) == set(SNAPSHOT_KEYS)
        assert set(actual.files) == set(SNAPSHOT_KEYS)
        assert not np.array_equal(actual["iglm"], actual["ema"]), (
            "synthetic snapshot must exercise iGLM polynomial detrending"
        )

        for key in SNAPSHOT_KEYS:
            np.testing.assert_allclose(
                actual[key],
                expected[key],
                rtol=0,
                atol=1e-8,
                equal_nan=True,
                err_msg=f"snapshot mismatch for {key}",
            )

if __name__ == "__main__":
    pytest.main()
