"""Run rtcaps_matcher.py in preproc mode with --snapshot flag to compare to known values using PILOT03_Run01.300vols.nii.
Could be automated in the future for quick testing after updates.
Note: Data_kalman has some nonnumeric values for some reason (nans maybe?) so np.load won't load if it's in there. Need to look into that."""

import os.path as osp
import pytest
import numpy as np

from rtcog.paths import DATA_DIR, OUTPUT_DIR

def test_snapshot():
    """New version of software vs old version (v2.0, using rtcaps_matcher.py)"""
    orig = osp.join(DATA_DIR, 'snapshot_all-on_snapshots.npz')
    res = osp.join(OUTPUT_DIR, 'new_snapshots.npz')

    with np.load(orig, allow_pickle=True) as f1, np.load(res, allow_pickle=True) as f2:

        assert np.array_equal(f1["Data_EMA"], f2["ema"])
        assert np.array_equal(f1["Data_iGLM"], f2["iglm"])
        assert np.array_equal(f1["Data_smooth"], f2["smooth"])
        assert np.array_equal(f1["Data_norm"], f2["snorm"])

if __name__ == "__main__":
    pytest.main()