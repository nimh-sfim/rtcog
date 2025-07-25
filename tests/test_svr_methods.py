import pytest
import numpy as np

from rtfmri.paths import CAP_labels
from rtfmri.utils.svr_methods import is_hit_rt01


def test_is_hit_rt01_two_above_thr(caplog):
    """Two templates meet criteria for match, so must return None"""
    svrscores = np.array([
        [0.0, 0.14],
        [0.2, 0.4],
        [0.5, 0.7],
        [0.9, 0.9],
        [0.0, 0.14],
        [0.2, 0.4],
        [0.3, 0.1]
    ])
    t = 1
    hit_thr = 0.5
    nconsec_vols = 2
    
    assert not is_hit_rt01(t, CAP_labels, svrscores, hit_thr, nconsec_vols)
    assert f' === is_hit_rt01 - nmatches 2' in caplog.text
  

def test_is_hit_rt01_one_match(caplog):
    """One template meets criteria for match"""
    svrscores = np.array([
        [0.0, 0.14],
        [0.2, 0.4],
        [0.3, 0.3],
        [0.5, 0.9],
        [0.0, 0.14],
        [0.2, 0.4],
        [0.3, 0.1]
    ])
    t = 1
    hit_thr = 0.5
    nconsec_vols = 2
    
    assert is_hit_rt01(t, CAP_labels, svrscores, hit_thr, nconsec_vols) == 'Audi'
    assert f' === is_hit_rt01 - nmatches 1' in caplog.text

    
def test_is_hit_rt01_no_match(caplog):
    """Nothing is above the threshold"""
    svrscores = np.array([
        [0.0, 0.14],
        [0.2, 0.4],
        [0.3, 0.3],
        [0.9, 0.4],
        [0.0, 0.14],
        [0.2, 0.4],
        [0.3, 0.1]
    ])
    t = 1
    hit_thr = 0.5
    nconsec_vols = 2
    
    assert not is_hit_rt01(t, CAP_labels, svrscores, hit_thr, nconsec_vols)
    assert f' === is_hit_rt01 - nmatches 0' in caplog.text

    
def test_is_hit_rt01_no_match_by_nconsec(caplog):
    """One template is above threshold, but only for this TR"""
    svrscores = np.array([
        [0.0, 0.14],
        [0.2, 0.4],
        [0.3, 0.3],
        [0.4, 0.9],
        [0.0, 0.14],
        [0.2, 0.4],
        [0.3, 0.1]
    ])
    t = 1
    hit_thr = 0.5
    nconsec_vols = 2
    
    assert not is_hit_rt01(t, CAP_labels, svrscores, hit_thr, nconsec_vols)
    assert f' === is_hit_rt01 - nmatches 1' in caplog.text


if __name__ == "__main__":
    pytest.main()