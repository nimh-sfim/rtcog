import numpy as np
import pytest

from rtcog.matching.matching_utils import nmi_bin_data, nmi_from_bins, nmi_n_bins


def test_nmi_n_bins_uses_ceiling_cube_root():
    assert nmi_n_bins(8) == 2
    assert nmi_n_bins(9) == 3
    assert nmi_n_bins(257) == 7


def test_nmi_bin_data_uses_expected_range():
    data = np.arange(8, dtype=float)

    bins = nmi_bin_data(data, n_bins=2)

    assert bins.shape == data.shape
    assert bins.min() == 1
    assert bins.max() == 2


def test_nmi_bin_data_constant_vector_uses_single_bin():
    bins = nmi_bin_data(np.ones(5), n_bins=3)

    np.testing.assert_array_equal(bins, np.ones(5, dtype=np.int16))


def test_nmi_bin_data_rejects_empty_vector():
    with pytest.raises(ValueError, match="empty"):
        nmi_bin_data([])


def test_nmi_identical_bins_is_high_similarity():
    data = np.arange(8, dtype=float)
    bins = nmi_bin_data(data, n_bins=2)

    score = nmi_from_bins(bins, bins, n_bins=2) - 1

    assert np.isclose(score, 1.0)


def test_nmi_from_bins_rejects_empty_vectors():
    with pytest.raises(ValueError, match="empty"):
        nmi_from_bins([], [])
