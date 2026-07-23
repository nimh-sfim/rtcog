import numpy as np
import pytest

from rtcog.matching.matching_utils import (
    nmi_bin_data,
    nmi_from_bins,
    nmi_n_bins,
    pearson_correlations,
)


def test_pearson_correlations_scores_multiple_templates():
    templates = np.array([
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [0, 1, 0, 1],
    ], dtype=np.float32)
    template_centered = templates - templates.mean(axis=1, keepdims=True)
    template_norms = np.linalg.norm(template_centered, axis=1)

    correlations = pearson_correlations(
        [0, 1, 2, 3],
        template_centered,
        template_norms,
    )

    np.testing.assert_allclose(correlations, [1, -1, 0.4472136], rtol=1e-6)


def test_pearson_correlations_returns_zero_for_degenerate_inputs():
    templates = np.array([[1, 1, 1], [0, 1, 2]], dtype=np.float32)
    template_centered = templates - templates.mean(axis=1, keepdims=True)
    template_norms = np.linalg.norm(template_centered, axis=1)

    np.testing.assert_array_equal(
        pearson_correlations([2, 2, 2], template_centered, template_norms),
        [0, 0],
    )
    np.testing.assert_array_equal(
        pearson_correlations([0, 1, 2], template_centered, template_norms),
        [0, 1],
    )
    np.testing.assert_array_equal(
        pearson_correlations([0, np.nan, 2], template_centered, template_norms),
        [0, 0],
    )


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
