import logging
import pytest
import numpy as np

from rtcog.matching.hit_detector import HitDetector
from rtcog.matching.hit_opts import HitOpts

@pytest.fixture
def labels():
    return ['VPol','DMN','SMot','Audi','ExCn','rFPa','lFPa']

@pytest.fixture
def hit_detector():
    hit_opts = HitOpts(
        nconsec_vols=2,
        hit_thr=0.5,
        nonline=1,
        do_mot=True,
        mot_thr=0.2
    )
    return HitDetector(hit_opts)

@pytest.fixture(autouse=True)
def enable_debug_logging(caplog):
    caplog.set_level(logging.DEBUG, logger="GENERAL")

def test_is_hit_two_above_thr(hit_detector, labels, caplog):
    """Two templates meet criteria for match, so must return None"""
    hit_detector = hit_detector
    labels = labels
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
    assert not hit_detector.is_hit(t, labels, svrscores)
    assert f' === is_hit - nmatches 2' in caplog.text
  
def test_is_hit_one_match(hit_detector, labels, caplog):
    """One template meets criteria for match"""
    hit_detector = hit_detector
    labels = labels
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

    assert hit_detector.is_hit(t, labels, svrscores) == 'Audi'
    assert f' === is_hit - nmatches 1' in caplog.text

def test_is_hit_no_match(hit_detector, labels, caplog):
    """Nothing is above the threshold"""
    hit_detector = hit_detector
    labels = labels
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
    
    assert not hit_detector.is_hit(t, labels, svrscores)
    assert f' === is_hit - nmatches 0' in caplog.text

def test_is_hit_no_match_by_nconsec(hit_detector, labels, caplog):
    """One template is above threshold, but only for this TR"""
    hit_detector = hit_detector
    labels = labels
    labels = labels
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
    
    assert not hit_detector.is_hit(t, labels, svrscores)
    assert f' === is_hit - nmatches 1' in caplog.text

def test_is_hit_no_match_by_nconsec(hit_detector, labels, caplog):
    """One template is above threshold, but only for this TR"""
    hit_detector = hit_detector
    labels = labels
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
    
    assert not hit_detector.is_hit(t, labels, svrscores)
    assert f' === is_hit - nmatches 1' in caplog.text


def test_is_hit_nonline_of_two(hit_detector, labels, caplog):
    """Two templates are online with nonline of two. Winner should take all."""
    hit_detector = hit_detector
    labels = labels
    svrscores = np.array([
        [0.0, 0.14],
        [0.2, 0.4],
        [0.5, 0.9],
        [0.9, 0.8],
        [0.0, 0.14],
        [0.2, 0.4],
        [0.3, 0.1]
    ])
    t = 1
    hit_detector.nonline = 2
    
    assert hit_detector.is_hit(t, labels, svrscores) == 'SMot'
    assert f' === is_hit - nmatches 2' in caplog.text


def test_is_hit_nonline_of_two_with_three_over_thr(hit_detector, labels, caplog):
    """Three templates are online, but only two are allowed"""
    hit_detector = hit_detector
    labels = labels
    svrscores = np.array([
        [0.0, 0.14],
        [0.2, 0.4],
        [0.5, 0.7],
        [0.9, 0.9],
        [0.0, 0.14],
        [0.6, 0.6],
        [0.3, 0.1]
    ])
    t = 1
    hit_detector.nonline = 2
    
    assert not hit_detector.is_hit(t, labels, svrscores)
    assert f' === is_hit - nmatches 3' in caplog.text

def test_is_hit_no_match_by_nconsec_nonline_of_two(hit_detector, labels, caplog):
    """Two templates are above threshold, but only for this TR"""
    hit_detector = hit_detector
    labels = labels
    svrscores = np.array([
        [0.0, 0.14],
        [0.2, 0.4],
        [0.3, 0.8],
        [0.4, 0.9],
        [0.0, 0.14],
        [0.2, 0.4],
        [0.3, 0.1]
    ])
    t = 1
    hit_detector.nonline = 2
    
    assert not hit_detector.is_hit(t, labels, svrscores)
    assert f' === is_hit - nmatches 2' in caplog.text
