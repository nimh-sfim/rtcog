import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, mock_open

from rtcog.matching.matcher import Matcher, SVRMatcher, MaskMatcher, NMIMatcher
from rtcog.matching.matching_opts import MatchingOpts

# ----------------------
# Fixtures and Mocks
# ----------------------
@pytest.fixture
def match_opts():
    return MatchingOpts(match_method="mask", match_start=0, vols_noaction=4)


# ----------------------
# Matcher Base Tests
# ----------------------
def test_matcher_registry_addition():
    class TestMatcher(Matcher):
        pass

    assert "test" in Matcher.registry
    assert Matcher.registry["test"] is TestMatcher

def test_match_from_name_unknown():
    with pytest.raises(ValueError):
        Matcher.from_name("nonexistent")

def test_match_scores_shape(make_sync_mock, match_opts):
    sync_events = make_sync_mock()
    class DummyMatcher(Matcher):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.Ntemplates = 2
            self.shared_arr = np.zeros((2, 5))
        def _match(self, tr_data):
            return np.array([0.1, 0.2])

    m = DummyMatcher(match_opts, Nt=5, sync=sync_events, match_path=None)
    scores = m.match(t=0, n=0, tr_data=np.zeros((10,)))
    assert scores.shape == (2, 5)
    sync_events.new_tr.set.assert_called_once()

def test_match_invalid_shape(make_sync_mock, match_opts):
    sync_events = make_sync_mock()
    class DummyMatcher(Matcher):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.Ntemplates = 2
            self.shared_arr = np.zeros((2, 5))
        def _match(self, tr_data):
            return np.zeros((2,2))  # Invalid shape

    m = DummyMatcher(match_opts, Nt=5, sync=sync_events, match_path=None)
    with pytest.raises(ValueError):
        m.match(t=0, n=0, tr_data=np.zeros((10,)))


def test_match_invalid_score_count(make_sync_mock, match_opts):
    sync_events = make_sync_mock()
    class DummyMatcher(Matcher):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.Ntemplates = 2
            self.shared_arr = np.zeros((2, 5))
        def _match(self, tr_data):
            return np.array([0.1])

    m = DummyMatcher(match_opts, Nt=5, sync=sync_events, match_path=None)
    with pytest.raises(ValueError, match="expected 2"):
        m.match(t=0, n=0, tr_data=np.zeros((10,)))

# ----------------------
# SVRMatcher Tests
# ----------------------
@patch("builtins.open", new_callable=mock_open)
@patch("pickle.load")
@patch.object(SVRMatcher, "setup_shared_memory")
def test_svrmatcher_loads_model(mock_setup_shm, mock_pickle_load, mock_file, match_opts, make_sync_mock):
    sync_events = make_sync_mock()
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.5]
    mock_pickle_load.return_value = {"template1": mock_model}

    matcher = SVRMatcher(match_opts, Nt=5, sync=sync_events, match_path="fake_path.pkl")
    assert matcher.Ntemplates == 1
    assert matcher.template_labels == ["template1"]
    mock_setup_shm.assert_called_once()
    sync_events.shm_ready.set.assert_called_once()

def test_svrmatcher_match_logic(match_opts):
    class DummyModel:
        def predict(self, X):
            return [np.sum(X)]
    
    matcher = SVRMatcher.__new__(SVRMatcher)
    matcher.action_start = 0
    matcher.Nt = 3
    matcher.Ntemplates = 2
    matcher.template_labels = ["a","b"]
    matcher.input = {"a": DummyModel(), "b": DummyModel()}
    matcher.shared_arr = np.zeros((2,3))
    matcher.mp_new_tr = MagicMock()

    tr_data = np.array([1,2,3])
    scores = matcher._match(tr_data)
    assert scores.shape == (2,)
    np.testing.assert_allclose(scores, [6,6])

# ----------------------
# MaskMatcher Tests
# ----------------------
@patch("numpy.load")
@patch.object(MaskMatcher, "setup_shared_memory")
def test_maskmatcher_loads_file(mock_setup_shm, mock_np_load, match_opts, make_sync_mock):
    sync_events = make_sync_mock()
    mock_np_load.return_value = {
        "labels": ["a"],
        "masked_templates": np.array({"a": np.array([1,2])}),
        "masks": np.array({"a": np.array([True, False])}),
        "voxel_counts": np.array({"a": 1})
    }

    matcher = MaskMatcher(match_opts, Nt=5, sync=sync_events, match_path="fake.npy")
    assert matcher.Ntemplates == 1
    assert matcher.template_labels == ["a"]
    mock_setup_shm.assert_called_once()
    sync_events.shm_ready.set.assert_called_once()

def test_maskmatcher_match_logic(match_opts):
    matcher = MaskMatcher.__new__(MaskMatcher)
    matcher.Ntemplates = 1
    matcher.template_labels = ["a"]
    matcher.masked_templates = {"a": np.array([1])}
    matcher.masks = {"a": np.array([True, False])}
    matcher.voxel_counts = {"a": 1}

    tr_data = np.array([[10,20]])
    scores = matcher._match(tr_data)
    assert scores.shape == (1,)
    assert np.isclose(scores[0], 10)


# ----------------------
# NMIMatcher Tests
# ----------------------
@patch.object(NMIMatcher, "setup_shared_memory")
def test_nmimatcher_loads_file(mock_setup_shm, tmp_path, match_opts, make_sync_mock):
    sync_events = make_sync_mock()
    match_path = tmp_path / "nmi_templates.npz"
    np.savez(
        match_path,
        labels=np.array(["ac001"]),
        templates=np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.float32),
        n_bins=np.array(2),
    )

    matcher = NMIMatcher(match_opts, Nt=5, sync=sync_events, match_path=str(match_path))

    assert matcher.Ntemplates == 1
    assert matcher.template_labels == ["ac001"]
    assert matcher.template_bins.shape == (1, 8)
    mock_setup_shm.assert_called_once()
    sync_events.shm_ready.set.assert_called_once()


@patch.object(NMIMatcher, "setup_shared_memory")
def test_nmimatcher_accepts_precomputed_bins_with_raw_templates(mock_setup_shm, tmp_path, match_opts, make_sync_mock):
    sync_events = make_sync_mock()
    match_path = tmp_path / "nmi_templates.npz"
    precomputed_bins = np.array([[1, 1, 1, 2, 2, 2, 2, 2]], dtype=np.int16)
    np.savez(
        match_path,
        labels=np.array(["ac001"]),
        templates=np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.float32),
        template_bins=precomputed_bins,
        n_bins=np.array(2),
    )

    matcher = NMIMatcher(match_opts, Nt=5, sync=sync_events, match_path=str(match_path))

    np.testing.assert_array_equal(matcher.template_bins, precomputed_bins)
    mock_setup_shm.assert_called_once()
    sync_events.shm_ready.set.assert_called_once()


@patch.object(NMIMatcher, "setup_shared_memory")
def test_nmimatcher_rejects_bins_without_raw_templates(mock_setup_shm, tmp_path, match_opts, make_sync_mock):
    sync_events = make_sync_mock()
    match_path = tmp_path / "nmi_bins_only.npz"
    np.savez(
        match_path,
        labels=np.array(["ac001"]),
        template_bins=np.array([[1, 1, 1, 1, 2, 2, 2, 2]], dtype=np.int16),
        n_bins=np.array(2),
    )

    with pytest.raises(ValueError, match='raw "templates"'):
        NMIMatcher(match_opts, Nt=5, sync=sync_events, match_path=str(match_path))

    mock_setup_shm.assert_not_called()
    sync_events.end.set.assert_called_once()


@pytest.mark.parametrize(
    ("labels", "templates"),
    [
        (["ac001"], np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32)),
        (["ac001", "ac002"], np.arange(16, dtype=np.float32).reshape(8, 2)),
    ],
)
@patch.object(NMIMatcher, "setup_shared_memory")
def test_nmimatcher_rejects_noncanonical_template_shape(
    mock_setup_shm,
    labels,
    templates,
    tmp_path,
    match_opts,
    make_sync_mock,
):
    sync_events = make_sync_mock()
    match_path = tmp_path / "nmi_bad_templates.npz"
    np.savez(
        match_path,
        labels=np.array(labels),
        templates=templates,
        n_bins=np.array(2),
    )

    with pytest.raises(ValueError, match="must have shape"):
        NMIMatcher(match_opts, Nt=5, sync=sync_events, match_path=str(match_path))

    mock_setup_shm.assert_not_called()
    sync_events.end.set.assert_called_once()


def test_nmimatcher_returns_signed_similarity(match_opts):
    matcher = NMIMatcher.__new__(NMIMatcher)
    matcher.Ntemplates = 2
    matcher.template_labels = ["positive", "negative"]
    matcher.templates = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [7, 6, 5, 4, 3, 2, 1, 0],
    ], dtype=np.float32)
    matcher.template_bins = np.array([
        [1, 1, 1, 1, 2, 2, 2, 2],
        [2, 2, 2, 2, 1, 1, 1, 1],
    ], dtype=np.int16)
    matcher.template_centered = matcher.templates - matcher.templates.mean(axis=1, keepdims=True)
    matcher.template_norms = np.linalg.norm(matcher.template_centered, axis=1)
    matcher.Nvoxels = 8
    matcher.n_bins = 2

    scores = matcher._match(np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32))

    assert scores.shape == (2,)
    assert np.isclose(scores[0], 1.0)
    assert np.isclose(scores[1], -1.0)
