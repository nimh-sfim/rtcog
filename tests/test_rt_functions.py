import pytest
import numpy as np
from numpy.testing import assert_array_equal

from rtfmri.preproc.preproc_utils import gen_polort_regressors, _is_pos_def, _iGLMVol, rt_regress_vol
from rtfmri.preproc.preproc_utils import rt_EMA_vol
from rtfmri.preproc.preproc_utils import init_kalman, _kalman_filter, rt_kalman_vol
from rtfmri.preproc.preproc_utils import rt_smooth_vol, rt_snorm_vol


def test_gen_polort_regressors():
    result = gen_polort_regressors(3, 5)
    assert result.shape == (5, 3)


def test_is_pos_def():
    pos_def_mat = np.array([[1, 0], [0, 1]])
    assert _is_pos_def(pos_def_mat)

    not_pos_def_mat = np.array([[1, 2], [3, 4]])
    assert not _is_pos_def(not_pos_def_mat)


def test_iGLMVol():
    n = 10
    Yn = np.random.randn(5, 1)
    Fn = np.random.randn(3, 1)
    Dn = np.zeros((5, 3))
    Cn = np.zeros((3, 3))
    s2n = np.zeros((5, 1))
    
    Bn, Cn_updated, Dn_updated, s2n_updated = _iGLMVol(n, Yn, Fn, Dn, Cn, s2n)
    
    assert Bn.shape == (5, 3)
    assert Cn_updated.shape == (3, 3)
    assert Dn_updated.shape == (5, 3)
    assert s2n_updated.shape == (5, 1)


def test_rt_regress_vol():
    n = 5
    Yn = np.random.randn(5, 1)
    Fn = np.random.randn(3, 1)
    prev = {'Cn': np.zeros((3, 3)), 'Dn': np.zeros((5, 3)), 's2n': np.zeros((5, 1))}

    Yn_d, new, Bn = rt_regress_vol(n, Yn, Fn, prev)
    
    assert Yn_d.shape == (5, 1)
    assert new['Cn'].shape == (3, 3)


def test_rt_EMA_vol_do_operation_first():
    n = 1
    th = 0.98
    data = np.array([[1, 2, 3], [4, 5, 6]])
    filt_in = None
    
    data_out, filt_out = rt_EMA_vol(n, th, data, filt_in, do_operation=True)
    
    assert len(data_out) == 2
    assert len(filt_out) == 2


def test_rt_EMA_vol_do_operation_second():
    n = 2
    th = 0.98
    data = np.array([[1, 2, 3], [4, 5, 6]])
    filt_in = [[1], [2]]
    
    data_out, filt_out = rt_EMA_vol(n, th, data, filt_in, do_operation=True)
    
    assert len(data_out) == 2
    assert len(filt_out) == 2


def test_rt_EMA_vol_no_operation():
    n = 2
    th = 0.98
    data = np.array([[1, 2, 3], [4, 5, 6]])
    filt_in = None
    
    data_out, filt_out = rt_EMA_vol(n, th, data, filt_in, do_operation=False)
    
    assert_array_equal(data_out, np.array([[3], [6]]))
    assert filt_out is None


# def test_rt_EMA_vol_real_data(sample_data):
#     n = 1
#     th = 0.98
#     data = np.array([[1, 2, 3], [4, 5, 6]])
#     filt_in = None
    
#     data_out, filt_out = rt_EMA_vol(n, th, data, filt_in, do_operation=True)
    
#     assert len(data_out) == 2
#     assert len(filt_out) == 2


def test_init_kalman():
    res = init_kalman(2, 2)
    
    for i, x in enumerate(res):
        if not np.array_equal(x, np.array([[0.0, 0.0], [0.0, 0.0]])):
            raise AssertionError(f'Matrix at index {i} did not match the expected result. '
                                 f'Expected:\n[[0.0, 0.0], [0.0, 0.0]]\n'
                                 f'Got:\n{x}')


def test_kalman_filter():
    pass


def test_kalman_filter_mv():
    pass


def test_rt_kalman_vol_no_kalman(sample_data):
    res = rt_kalman_vol(
        sample_data.n,
        sample_data.t,
        sample_data.this_t_data.reshape(-1, 1),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        do_operation=False
    )

    assert_array_equal(sample_data.this_t_data, np.squeeze(res[0]))
    assert all(x is None for x in res[1:])


def test_rt_smooth_vol(sample_data):
    res = rt_smooth_vol(sample_data.this_t_data, sample_data.mask_img, fwhm=4)

    assert isinstance(res, np.ndarray)
    assert res.ndim == 2
    assert res.shape == (sample_data.this_t_data.shape[0], 1)

    assert not np.allclose(res[:, 0], sample_data.this_t_data)
    

def test_rt_smooth_vol_no_operation(sample_data):
    res = rt_smooth_vol(sample_data.this_t_data, sample_data.mask_img, do_operation=False)

    assert isinstance(res, np.ndarray)
    assert res.ndim == 2
    assert res.shape == (sample_data.this_t_data.shape[0], 1)

    assert res.all() == sample_data.this_t_data[:, np.newaxis].all()


def test_rt_snorm_vol(sample_data):
    res = rt_snorm_vol(sample_data.this_t_data)

    assert res.shape[0] == (sample_data.this_t_data.shape[0])
    assert np.isclose(np.mean(res), 0.0)
    assert np.isclose(np.std(res), 1.0)


def test_rt_snorm_vol_no_operation(sample_data):
    res = rt_snorm_vol(sample_data.this_t_data, do_operation=False)

    assert res.shape[0] == (sample_data.this_t_data.shape[0])
    assert np.array_equal(res.flatten(), sample_data.this_t_data)    


if __name__ == "__main__":
    pytest.main()