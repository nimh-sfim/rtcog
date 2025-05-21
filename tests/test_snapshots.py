"""Run rtcaps_matcher.py in preproc mode with --snapshot flag to compare to known values using PILOT03_Run01.300vols.nii.
Could be automated in the future for quick testing after updates.
Note: Data_kalman has some nonnumeric values for some reason (nans maybe?) so np.load won't load if it's in there. Need to look into that."""

import os.path as osp
import pytest
import numpy as np

from rtcaps.config import DATA_DIR, OUTPUT_DIR

# @pytest.mark.snapshot
def test_snapshot():
    orig = osp.join(DATA_DIR, 'snapshot_all-on_snapshots.npz')
    # res = osp.join(OUTPUT_DIR, 'snapshot_all-on_test_snapshots.npz')
    res = osp.join(OUTPUT_DIR, 'module_snapshots.npz')

    with np.load(orig, allow_pickle=True) as f1, np.load(res, allow_pickle=True) as f2:
        # assert set(f1.files) == set(f2.files)

        for key in f1.files:
            a1 = f1[key]
            a2 = f2[key]

            assert np.allclose(a1, a2)

if __name__ == "__main__":
    pytest.main()