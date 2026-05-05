"""
Tests for rtcog/processor/basic_processor.py
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from rtcog.processor.basic_processor import BasicProcessor


@patch('rtcog.processor.basic_processor.set_logger')
@patch('rtcog.processor.basic_processor.load_fMRI_file')
@patch('rtcog.processor.basic_processor.Pipeline')
def test_init_with_valid_options(mock_pipeline, mock_load, mock_set_logger, make_sync_mock):
    """Test BasicProcessor initialization with valid options."""
    mock_options = MagicMock()
    mock_options.exp_type = 'basic'
    mock_options.nvols = 100
    mock_options.tr = 2.0
    mock_options.discard = 5
    mock_options.mask_path = '/path/to/mask.nii'
    mock_options.debug = False
    mock_options.silent = False

    mock_sync = make_sync_mock()

    mock_mask_img = MagicMock()
    mock_mask_img.get_fdata.return_value = np.ones((10, 10, 10))
    mock_load.return_value = mock_mask_img

    mock_logger = MagicMock()
    mock_set_logger.return_value = mock_logger

    processor = BasicProcessor(mock_options, mock_sync)

    assert processor.exp_type == 'basic'
    assert processor.Nt == 100
    assert processor.nvols_discard == 5
    assert processor.mask_Nv == 1000  # 10*10*10
    mock_load.assert_called_once_with('/path/to/mask.nii')
    mock_pipeline.assert_called_once_with(mock_options, 100, 1000, mock_mask_img)


@patch('rtcog.processor.basic_processor.set_logger')
def test_init_no_mask_path(mock_set_logger, make_sync_mock):
    """Test BasicProcessor initialization without mask path."""
    mock_options = MagicMock()
    mock_options.mask_path = None
    mock_options.debug = False
    mock_options.silent = False

    mock_sync = make_sync_mock()

    mock_logger = MagicMock()
    mock_set_logger.return_value = mock_logger

    with pytest.raises(ValueError, match="No mask was provided!"):
        BasicProcessor(mock_options, mock_sync)
    mock_sync.end.set.assert_called_once()


@patch('rtcog.processor.basic_processor.set_logger')
def test_init_zero_nvols_sets_end(mock_set_logger, make_sync_mock):
    """Test BasicProcessor rejects a run with no expected volumes."""
    mock_options = MagicMock()
    mock_options.exp_type = 'basic'
    mock_options.nvols = 0
    mock_options.discard = 5
    mock_options.mask_path = '/path/to/mask.nii'
    mock_options.debug = False
    mock_options.silent = False

    mock_sync = make_sync_mock()

    with pytest.raises(ValueError, match="Number of expected volumes"):
        BasicProcessor(mock_options, mock_sync)
    mock_sync.end.set.assert_called_once()


@patch('rtcog.processor.basic_processor.set_logger')
@patch('rtcog.processor.basic_processor.load_fMRI_file')
def test_init_empty_mask_sets_end(mock_load, mock_set_logger, make_sync_mock):
    """Test BasicProcessor rejects an empty mask."""
    mock_options = MagicMock()
    mock_options.exp_type = 'basic'
    mock_options.nvols = 100
    mock_options.discard = 5
    mock_options.mask_path = '/path/to/mask.nii'
    mock_options.debug = False
    mock_options.silent = False

    mock_mask_img = MagicMock()
    mock_mask_img.get_fdata.return_value = np.zeros((2, 2, 2))
    mock_load.return_value = mock_mask_img

    mock_sync = make_sync_mock()

    with pytest.raises(ValueError, match="Provided mask file is empty"):
        BasicProcessor(mock_options, mock_sync)
    mock_sync.end.set.assert_called_once()


@patch('rtcog.processor.basic_processor.set_logger')
@patch('rtcog.processor.basic_processor.load_fMRI_file', side_effect=OSError("boom"))
def test_init_mask_load_error_sets_end(mock_load, mock_set_logger, make_sync_mock):
    """Test BasicProcessor converts mask-load failures into runtime errors."""
    mock_options = MagicMock()
    mock_options.exp_type = 'basic'
    mock_options.nvols = 100
    mock_options.discard = 5
    mock_options.mask_path = '/path/to/mask.nii'
    mock_options.debug = False
    mock_options.silent = False

    mock_sync = make_sync_mock()

    with pytest.raises(RuntimeError, match="Error loading mask file: boom"):
        BasicProcessor(mock_options, mock_sync)
    mock_sync.end.set.assert_called_once()


@patch('rtcog.processor.basic_processor.set_logger')
@patch('rtcog.processor.basic_processor.load_fMRI_file')
@patch('rtcog.processor.basic_processor.Pipeline')
def test_compute_TR_data_impl(mock_pipeline, mock_load, mock_set_logger, make_sync_mock):
    """Test _compute_TR_data_impl processing."""
    mock_options = MagicMock()
    mock_options.exp_type = 'basic'
    mock_options.nvols = 100
    mock_options.tr = 2.0
    mock_options.discard = 5
    mock_options.mask_path = '/path/to/mask.nii'

    mock_sync = make_sync_mock()

    mock_mask_img = MagicMock()
    mock_mask_img.get_fdata.return_value = np.ones((10, 10, 10))
    mock_load.return_value = mock_mask_img

    mock_pipe = MagicMock()
    mock_pipeline.return_value = mock_pipe
    mock_pipe.process.return_value = np.array([1, 2, 3])

    processor = BasicProcessor(mock_options, mock_sync)

    motion = [[1], [2], [3], [4], [5], [6]]  # 6 params, each with values for TRs, but for t=0, [1,2,3,4,5,6]
    extra = [[0.1], [0.2], [0.3]]  # voxel data for TRs

    result = processor._compute_TR_data_impl(motion, extra)

    assert processor.t == 0
    assert processor.n == 0  # since t=0 < discard-1=4
    assert np.array_equal(result, np.array([1, 2, 3]))
    mock_pipe.process.assert_called_once()


@patch('rtcog.processor.basic_processor.set_logger')
@patch('rtcog.processor.basic_processor.load_fMRI_file')
@patch('rtcog.processor.basic_processor.Pipeline')
def test_compute_TR_data(mock_pipeline, mock_load, mock_set_logger, make_sync_mock):
    """Test compute_TR_data wrapper."""
    mock_options = MagicMock()
    mock_options.exp_type = 'basic'
    mock_options.nvols = 100
    mock_options.tr = 2.0
    mock_options.discard = 5
    mock_options.mask_path = '/path/to/mask.nii'

    mock_sync = make_sync_mock()

    mock_mask_img = MagicMock()
    mock_mask_img.get_fdata.return_value = np.ones((10, 10, 10))
    mock_load.return_value = mock_mask_img

    mock_pipe = MagicMock()
    mock_pipeline.return_value = mock_pipe

    mock_logger = MagicMock()
    mock_set_logger.return_value = mock_logger

    processor = BasicProcessor(mock_options, mock_sync)

    motion = [[1], [2], [3], [4], [5], [6]]
    extra = [[0.1], [0.2], [0.3]]

    result = processor.compute_TR_data(motion, extra)

    assert result == 1
    mock_logger.info.assert_called()


@patch('rtcog.processor.basic_processor.set_logger')
@patch('rtcog.processor.basic_processor.load_fMRI_file')
@patch('rtcog.processor.basic_processor.Pipeline')
def test_end_run(mock_pipeline, mock_load, mock_set_logger, make_sync_mock):
    """Test end_run method."""
    mock_options = MagicMock()
    mock_options.mask_path = '/path/to/mask.nii'

    mock_sync = make_sync_mock()

    mock_mask_img = MagicMock()
    mock_mask_img.get_fdata.return_value = np.ones((10, 10, 10))
    mock_load.return_value = mock_mask_img

    mock_pipe = MagicMock()
    mock_pipeline.return_value = mock_pipe

    processor = BasicProcessor(mock_options, mock_sync)

    processor.end_run(save=True)

    mock_pipe.final_steps.assert_called_once_with(save=True)
    mock_sync.end.set.assert_called_once()
