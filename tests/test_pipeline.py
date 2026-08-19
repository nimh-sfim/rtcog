import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from rtcog.preproc.pipeline import Pipeline
from rtcog.utils.exceptions import VolumeOverflowError


def make_pipeline_options(tmp_path=None):
    options = MagicMock()
    options.save_orig = False
    options.out_dir = str(tmp_path) if tmp_path is not None else "/tmp"
    options.out_prefix = "test"
    options.snapshot = False
    options.steps = []
    return options


def test_pipeline_imports_custom_steps_module():
    assert "rtcog.preproc.custom_steps" in sys.modules


@patch('rtcog.preproc.pipeline.PreprocStep')
def test_pipeline_build_steps_empty(mock_preproc_step):
    """Test build_steps with no steps."""
    mock_options = MagicMock()
    mock_options = MagicMock()
    mock_options.steps = [
        {"name": "test_step", "enabled": True, "param": "value"}
    ]
    mock_preproc_step.registry = {"test_step": MagicMock()}
    pipeline = Pipeline.__new__(Pipeline)  # Create without __init__
    pipeline.step_opts = mock_options.steps
    pipeline.step_registry = mock_preproc_step.registry
    pipeline.mask_Nv = 50
    pipeline.Nt = 100
    pipeline.steps = []

    mock_step_class = MagicMock()
    mock_step_instance = MagicMock()
    mock_step_instance.name = "test_step"
    mock_step_class.return_value = mock_step_instance
    mock_preproc_step.registry["test_step"] = mock_step_class

    pipeline.build_steps()

    assert len(pipeline.steps) == 1
    assert pipeline.steps[0] == mock_step_instance
    mock_step_class.assert_called_once_with(save=False, Nv=50, Nt=100, param="value")


@patch('rtcog.preproc.pipeline.PreprocStep')
def test_pipeline_build_steps_disabled(mock_preproc_step):
    """Test build_steps with disabled steps."""
    mock_options = MagicMock()
    mock_options.steps = [
        {"name": "test_step", "enabled": False}
    ]
    mock_preproc_step.registry = {"test_step": MagicMock()}
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.step_opts = mock_options.steps
    pipeline.step_registry = mock_preproc_step.registry
    pipeline.mask_Nv = 50
    pipeline.Nt = 100
    pipeline.steps = []

    pipeline.build_steps()

    assert len(pipeline.steps) == 0


@patch('rtcog.preproc.pipeline.PreprocStep')
def test_pipeline_build_steps_unknown(mock_preproc_step):
    """Test build_steps with unknown step."""
    mock_options = MagicMock()
    mock_options.steps = [
        {"name": "unknown_step", "enabled": True}
    ]
    mock_preproc_step.registry = {}
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.step_opts = mock_options.steps
    pipeline.step_registry = mock_preproc_step.registry
    pipeline.mask_Nv = 50
    pipeline.Nt = 100
    pipeline.steps = []

    with pytest.raises(SystemExit):
        pipeline.build_steps()


@patch('rtcog.preproc.pipeline.PreprocStep')
def test_pipeline_processed_tr_property(mock_preproc_step):
    """Test processed_tr property getter and setter."""
    mock_options = MagicMock()
    mock_options.steps = []
    pipeline = Pipeline(mock_options, 100, 50, None)

    result = pipeline.processed_tr
    assert np.array_equal(result, pipeline._processed_tr)

    new_value = np.ones((50, 1))
    pipeline.processed_tr = new_value
    assert np.array_equal(pipeline._processed_tr, new_value)


def test_pipeline_process_discard_volume_initializes_arrays():
    pipeline = Pipeline(make_pipeline_options(), Nt=2, mask_Nv=3, mask_img=None)

    result = pipeline.process(t=0, n=0, motion=[0] * 6, this_t_data=np.array([1, 2, 3]))

    assert result is None
    assert pipeline.Nv == 3
    np.testing.assert_array_equal(pipeline.Data_FromAFNI[:, 0], np.array([1, 2, 3]))
    np.testing.assert_array_equal(pipeline.Data_processed, np.zeros((3, 2)))


def test_pipeline_process_kept_volume_without_steps_returns_raw_column():
    pipeline = Pipeline(make_pipeline_options(), Nt=2, mask_Nv=3, mask_img=None)
    pipeline.process(t=0, n=0, motion=[0] * 6, this_t_data=np.array([1, 2, 3]))

    result = pipeline.process(t=1, n=1, motion=[0] * 6, this_t_data=np.array([4, 5, 6]))

    np.testing.assert_array_equal(result, np.array([[4], [5], [6]]))
    np.testing.assert_array_equal(pipeline.Data_processed[:, 1], np.array([4, 5, 6]))


def test_pipeline_process_raises_on_volume_overflow():
    pipeline = Pipeline(make_pipeline_options(), Nt=1, mask_Nv=3, mask_img=None)
    pipeline.process(t=0, n=0, motion=[0] * 6, this_t_data=np.array([1, 2, 3]))

    with pytest.raises(VolumeOverflowError):
        pipeline.process(t=1, n=1, motion=[0] * 6, this_t_data=np.array([4, 5, 6]))
