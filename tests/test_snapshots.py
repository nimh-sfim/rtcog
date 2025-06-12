"""Run rtcaps_matcher.py in preproc mode with --snapshot flag to compare to known values using PILOT03_Run01.300vols.nii.
Could be automated in the future for quick testing after updates.
Note: Data_kalman has some nonnumeric values for some reason (nans maybe?) so np.load won't load if it's in there. Need to look into that."""

import os.path as osp
import pytest
import numpy as np

from rtcaps.config import DATA_DIR, OUTPUT_DIR

def test_snapshot_old():
    orig = osp.join(DATA_DIR, 'snapshot_all-on_snapshots.npz')
    res = osp.join(OUTPUT_DIR, 'snapshots.npz')

    with np.load(orig, allow_pickle=True) as f1, np.load(res, allow_pickle=True) as f2:

        for key in f1.files:
            a1 = f1[key]
            a2 = f2[key]

            assert np.array_equal(a1, a2)
        assert np.array_equal(f1["Data_norm"], f2["Data_processed"])

if __name__ == "__main__":
    pytest.main()

def test_snapshot_new():
    orig = osp.join(DATA_DIR, 'snapshot_all-on_snapshots.npz')
    res = osp.join(OUTPUT_DIR, 'snapshots.npz')

    with np.load(orig, allow_pickle=True) as f1, np.load(res, allow_pickle=True) as f2:

        assert np.array_equal(f1["Data_EMA"], f2["ema"])
        assert np.array_equal(f1["Data_iGLM"], f2["iglm"])
        assert np.array_equal(f1["Data_smooth"], f2["smooth"])
        assert np.array_equal(f1["Data_norm"], f2["snorm"])
        assert np.array_equal(f1["Data_norm"], f2["Data_processed"])

if __name__ == "__main__":
    pytest.main()