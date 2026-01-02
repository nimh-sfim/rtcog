import os
import tempfile
import pytest
from unittest.mock import patch
from rtcog.utils.core import file_exists, SharedClock, setup_afni


def test_file_exists():
    """Test file_exists with existing file."""
    with tempfile.NamedTemporaryFile() as tmp:
        result = file_exists(tmp.name)
        assert result == tmp.name


def test_file_exists_not_found():
    """Test file_exists with non-existent file."""
    with pytest.raises(FileNotFoundError, match="File not found"):
        file_exists("/non/existent/file")


def test_shared_clock_now():
    clock = SharedClock()
    import time
    time.sleep(0.01)  # Small delay
    elapsed = clock.now()
    assert elapsed > 0
    assert isinstance(elapsed, float)


@patch('rtcog.utils.core.shutil.which')
@patch.dict(os.environ, {'READTHEDOCS': 'True'})
def test_setup_afni_readthedocs(mock_which):
    """Test setup_afni in ReadTheDocs mode."""
    result = setup_afni()
    assert result == (None, None)


@patch('rtcog.utils.core.shutil.which')
def test_setup_afni_no_afni(mock_which):
    mock_which.return_value = None
    with pytest.raises(RuntimeError, match="AFNI not found"):
        setup_afni()
