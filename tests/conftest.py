import sys
import os.path as osp
import pytest
import numpy as np
import pandas as pd
import nibabel as nib
from unittest.mock import MagicMock

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
        help="run snapshot tests (implies local data for those tests)",
    )
    parser.addoption(
        "--local-data", action="store_true", default=False, help="run tests that require local fixture data"
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
    skip_local_data = pytest.mark.skip(reason="need --local-data option to run")
    skip_snapshot = pytest.mark.skip(reason="need --snapshot option to run")
    include_local_data = config.getoption("--local-data")
    include_snapshots = config.getoption("--snapshot")

    for item in items:
        explicitly_selected = _item_was_explicitly_selected(config, item)
        is_snapshot = "snapshot" in item.keywords
        needs_local_data = (
            "local_data" in item.keywords
            or "sample_data" in getattr(item, "fixturenames", ())
        )

        if is_snapshot and not (include_snapshots or explicitly_selected):
            item.add_marker(skip_snapshot)

        local_data_enabled = (
            include_local_data
            or explicitly_selected
            or (is_snapshot and include_snapshots)
        )
        if needs_local_data and not local_data_enabled:
            item.add_marker(skip_local_data)


class SampleData:
    def __init__(self):
        self.orig_img = nib.load(osp.join(DATA_DIR, 'test_epi_data100+orig.BRIK.gz'))
        self.mask_img = nib.load(osp.join(DATA_DIR, 'GMribbon_R4Feed.nii'))

        self.orig_data, self.mask_data = self.orig_img.get_fdata(), self.mask_img.get_fdata()

        self.this_t_data = np.load(osp.join(DATA_DIR, 'this_t_data.npy'))

        self.Nv = len(self.this_t_data)
        self.t = self.orig_data.shape[-1]
        self.n = self.t - 10
    

@pytest.fixture(scope="session")
def sample_data():
    return SampleData()
