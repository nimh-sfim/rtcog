import pytest
import numpy as np
from numpy.testing import assert_array_equal

from rtcog.preproc.helpers.preproc_utils import gen_polort_regressors 
from rtcog.preproc.helpers.preproc_utils import rt_smooth_vol
from rtcog.preproc.helpers.preproc_utils import calculate_spc, CircularBuffer
from rtcog.preproc.helpers.iglm import iGLM


def test_gen_polort_regressors():
    result = gen_polort_regressors(3, 5)
    assert result.shape == (5, 3)


def test_iGLMVol():
    iglm = iGLM()
    n = 10
    Yn = np.random.randn(5, 1)
    Fn = np.random.randn(3, 1)
    iglm.Dn = np.zeros((5, 3))
    iglm.Cn = np.zeros((3, 3))
    iglm.s2n = np.zeros((5, 1))
    
    Yn_d, Bn = iglm.regress_vol(n, Yn, Fn)
    
    assert Yn_d.shape == (5, 1)
    assert Bn.shape == (5, 3, 1)
    assert iglm.Cn.shape == (3, 3)
    assert iglm.Dn.shape == (5, 3)


def test_rt_regress_vol():
    iglm = iGLM()
    n = 5
    Yn = np.random.randn(5, 1)
    Fn = np.random.randn(3, 1)

    # Set state for n=5
    iglm.Cn = np.zeros((3, 3))
    iglm.Dn = np.zeros((5, 3))
    iglm.s2n = np.zeros((5, 1))

    Yn_d, Bn = iglm.regress_vol(n, Yn, Fn)
    
    assert Yn_d.shape == (5, 1)
    assert Bn.shape == (5, 3, 1)
    assert iglm.Cn.shape == (3, 3)



@pytest.mark.local_data
def test_rt_smooth_vol(sample_data):
    data_2d = sample_data.this_t_data[:, np.newaxis]
    res = rt_smooth_vol(data_2d, sample_data.mask_img, fwhm=4)

    assert isinstance(res, np.ndarray)
    assert res.ndim == 2
    assert res.shape == (sample_data.this_t_data.shape[0], 1)

    assert not np.allclose(res[:, 0], sample_data.this_t_data)


def test_calculate_spc_with_mean_present():
    current = np.array([12.0, 8.0])
    baseline = np.array([10.0, 10.0])

    res = calculate_spc(current, baseline, mean_removed=False)

    np.testing.assert_allclose(res, np.array([[20.0], [-20.0]]))


def test_calculate_spc_with_mean_removed():
    current = np.array([2.0, -2.0])
    baseline = np.array([10.0, 10.0])

    res = calculate_spc(current, baseline, mean_removed=True)

    np.testing.assert_allclose(res, np.array([[20.0], [-20.0]]))


def test_circular_buffer_returns_ordered_window_after_full():
    buffer = CircularBuffer(Nv=2, size=3)

    assert buffer.update(np.array([[1], [10]])) is None
    assert buffer.update(np.array([[2], [20]])) is None

    first_window = buffer.update(np.array([[3], [30]]))
    np.testing.assert_array_equal(first_window, np.array([[1, 2, 3], [10, 20, 30]]))

    second_window = buffer.update(np.array([[4], [40]]))
    np.testing.assert_array_equal(second_window, np.array([[2, 3, 4], [20, 30, 40]]))
    

if __name__ == "__main__":
    pytest.main()
