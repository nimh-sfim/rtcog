import os.path as osp
from unittest.mock import MagicMock

import nibabel as nib
import numpy as np
import pytest

from rtcog.paths import DATA_DIR


@pytest.fixture
def make_sync_mock():
    def _make(**overrides):
        sync = MagicMock()
        sync.hit = MagicMock()
        sync.action_end = MagicMock()
        sync.end = MagicMock()
        sync.new_tr = MagicMock()
        sync.shm_ready = MagicMock()
        sync.server_ready = MagicMock()
        sync.tr_index = MagicMock()
        for name, value in overrides.items():
            setattr(sync, name, value)
        return sync
    return _make


def pytest_addoption(parser):
    parser.addoption(
        "--snapshot",
        action="store_true",
        default=False,
        help="run snapshot tests after generating their run output",
    )


def _item_was_explicitly_selected(config, item):
    """Return whether pytest was given this test file or one of its node IDs."""
    item_path = osp.realpath(str(item.path))
    for arg in config.args:
        requested_path = str(arg).split("::", 1)[0]
        if osp.isfile(requested_path) and osp.realpath(requested_path) == item_path:
            return True
    return False


def pytest_collection_modifyitems(config, items):
    skip_snapshot = pytest.mark.skip(reason="need --snapshot option to run")
    include_snapshots = config.getoption("--snapshot")

    for item in items:
        explicitly_selected = _item_was_explicitly_selected(config, item)
        is_snapshot = "snapshot" in item.keywords

        if is_snapshot and not (include_snapshots or explicitly_selected):
            item.add_marker(skip_snapshot)


class SyntheticFMRIData:
    """Committed mathematical phantom shared by fMRI-related tests."""

    def __init__(self):
        self.img = nib.load(osp.join(DATA_DIR, "synthetic_snapshot_input.nii.gz"))
        self.mask_img = nib.load(
            osp.join(DATA_DIR, "synthetic_snapshot_mask.nii.gz")
        )
        self.data = self.img.get_fdata()
        self.mask = self.mask_img.get_fdata() == 1
        self.n_voxels = int(np.count_nonzero(self.mask))
        self.n_volumes = self.data.shape[-1]


@pytest.fixture(scope="session")
def synthetic_fmri_data():
    return SyntheticFMRIData()
