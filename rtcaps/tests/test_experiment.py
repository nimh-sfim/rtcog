import sys
from unittest.mock import MagicMock
import pytest
import logging
import numpy as np
from numpy.testing import assert_array_equal

sys.path.append('../')
from bin.rtcaps_matcher import Experiment


@pytest.fixture
def setup_experiment():
    options = MagicMock()
    options.silent = False
    options.debug = False
    options.exp_type = 'preproc'
    options.no_proc_chair = False
    options.fullscreen = False
    options.screen = 0
    options.nvols = 100
    options.tr = 1.0
    options.n_cores = 1
    options.save_ema = False
    options.save_smooth = False
    options.save_kalman = False
    options.save_iglm = False 
    options.save_orig = False
    options.save_all = False
    options.do_EMA = False
    options.do_iGLM = False
    options.do_kalman = False
    options.do_smooth = False
    options.do_snorm = False
    options.discard = 10
    options.iGLM_polort = 2
    options.iGLM_motion = True
    options.mask_path = None
    options.out_dir = '/output'
    options.out_prefix = 'experiment_test'

    mp_evt_hit = MagicMock()
    mp_evt_end = MagicMock()
    mp_evt_qa_end = MagicMock()
    
    mp_evt_hit.is_set.return_value = False
    mp_evt_end.is_set.return_value = False
    mp_evt_qa_end.is_set.return_value = False
    
    exp = Experiment(options, mp_evt_hit, mp_evt_end, mp_evt_qa_end)
    
    return exp


def test_first_volume(setup_experiment):
    experiment = setup_experiment
    experiment.t = -1
    motion_data = [[1], [2], [3], [4], [5], [6]]
    extra_data = [[1], [2], [3]]

    rv = experiment.compute_TR_data(motion_data, extra_data)

    assert rv == 1
    assert experiment.Nv == 3
    assert experiment.Data_norm.shape == (3, 1)
    assert_array_equal(experiment.Data_norm, np.array([[0], [0], [0]]))
    assert_array_equal(experiment.Data_FromAFNI,np.array([[1], [2], [3]]))


def test_second_volume(setup_experiment):
    experiment = setup_experiment
    experiment.t = 0
    experiment.Nv = 3
    experiment.Data_FromAFNI = np.array([[1], [2], [3]]) 
    experiment.Data_norm = np.zeros((3, 1))
    motion_data = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]
    extra_data = [[1, 1], [2, 2], [3, 3]]

    rv = experiment.compute_TR_data(motion_data, extra_data)

    assert rv == 1
    assert experiment.n == 0
    assert experiment.Nv == 3
    assert_array_equal(experiment.Data_norm, np.array([[0, 0], [0, 0], [0, 0]]))
    assert_array_equal(experiment.Data_FromAFNI, np.array([[1, 1], [2, 2], [3, 3]]))


def test_incorrect_motion(setup_experiment, caplog):
    experiment = setup_experiment
    experiment.t = 0

    motion_data = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    extra_data = [[1, 1], [2, 2], [3, 3]]

    with caplog.at_level(logging.ERROR):
        try:
            experiment.compute_TR_data(motion_data, extra_data)
        except SystemExit:
            pass
    
    assert 'Motion not read in correctly.' in caplog.text
    assert 'Expected length: 6 | Actual length: 5' in caplog.text
    

def test_incorrect_extra(setup_experiment, caplog):
    experiment = setup_experiment
    experiment.t = 0
    experiment.Nv = 3

    motion_data = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]
    extra_data = [[1, 1], [2, 2]]

    with caplog.at_level(logging.ERROR):
        try:
            experiment.compute_TR_data(motion_data, extra_data)
        except SystemExit:
            pass
    
    assert 'Extra data not read in correctly.' in caplog.text
    assert 'Expected length: 3 | Actual length: 2' in caplog.text


def test_non_discarded_volume(setup_experiment):
    experiment = setup_experiment
    experiment.t = 9
    experiment.Nv = 3
    experiment.save_orig = True
    experiment.Data_FromAFNI = np.array([[i] * 10 for i in range(1, 4)])
    experiment.Data_norm = np.zeros((3, 1))
    experiment.welford_M = np.zeros(3)
    experiment.welford_S = np.zeros(3)
    experiment.welford_std = np.zeros(3)
    
    motion_data = [[i] * 11 for i in range(1, 7)]
    extra_data = [[i] * 11 for i in range(1, 4)]

    rv = experiment.compute_TR_data(motion_data, extra_data)

    assert rv == 1
    assert experiment.n == 1
    assert experiment.Nv == 3
    assert_array_equal(experiment.Data_FromAFNI, np.array(extra_data))


if __name__ == "__main__":
    pytest.main()