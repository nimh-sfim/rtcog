import os
import sys
import time
import os.path as osp
import shutil
from multiprocessing import Event, Value
from ctypes import c_int
from rtcog.utils.sync import SyncEvents
from rtcog.utils.log import get_logger

log = get_logger()

def file_exists(path):
    """Check if file exists."""
    if not osp.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    return path

def create_sync_events():
    """Create multiprocessing infrastructure."""
    return SyncEvents(
        new_tr=Event(),
        shm_ready=Event(),
        action_end=Event(),
        hit=Event(),
        end=Event(),
        server_ready=Event(),
        tr_index=Value(c_int, -1)
    )
    
class SharedClock:
    """
    Minimal clock for tracking elapsed time across processes.

    This clock is initialized at instantiation and provides a method for
    retrieving the current time relative to that start point.

    Attributes
    ----------
    _start_time : float
        The absolute time (in seconds) when the clock was created, based on `time.perf_counter()`.
    """
    def __init__(self):
        """
        Initialize the SharedClock and store the current time as the reference start point.
        """
        self._start_time = time.perf_counter()
      
    def now(self):
        """
        Get the current time relative to the clock's start time.

        Returns
        -------
        float
            Elapsed time in seconds since the clock was initialized.
        """
        return time.perf_counter() - self._start_time
      
def setup_afni():
    if os.environ.get("READTHEDOCS") == "True":
        log.warning("AFNI not loaded. Mocking imports for documentation build")
        return None, None
    afni_path = shutil.which('afni')

    if not afni_path:
        log.error('++ ERROR: AFNI not found in the system PATH')
        raise RuntimeError("AFNI not found")

    abin_path = osp.dirname(afni_path)
    sys.path.insert(1, abin_path)

    from afnipy import module_test_lib
    testlibs = ['signal', 'time']
    if module_test_lib.num_import_failures(testlibs):
        raise RuntimeError("AFNI module import failures")

    from realtime_receiver import ReceiverInterface
    from afnipy import lib_realtime as RT

    return ReceiverInterface, RT
