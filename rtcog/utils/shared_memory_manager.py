from typing import Optional
from multiprocessing.shared_memory import SharedMemory
from rtcog.utils.log import get_logger

log = get_logger()

class SharedMemoryManager:
    """
    Manager for SharedMemory with open and cleanup logic.

    This class wraps multiprocessing.shared_memory.SharedMemory to provide
    automatic cleanup.
    """

    def __init__(self, name: str, create: bool = False, size: Optional[int] = None):
        """
        Initialize the SharedMemoryManager.

        Parameters
        ----------
        name : str
            Name of the shared memory segment.
        create : bool, optional
            If True, create a new shared memory segment. If False, attach to existing.
        size : int, optional
            Size in bytes for creation (required if create=True).
        """
        self.name = name
        self.shm = None
        self._owner = create
        self._size = size

        if create and size is None:
            raise ValueError("size is required when create=True")

        log.debug(f'Initialized SharedMemoryManager for "{self.name}", create={self._owner}, size={self._size}')

    def open(self) -> SharedMemory:
        """Create/attach to shared memory."""
        try:
            if self._owner:
                try:
                    self.shm = SharedMemory(create=True, size=self._size, name=self.name)
                except FileExistsError:  # If already exists, clean up and retry
                    log.debug(f"Shared memory '{self.name}' already exists, cleaning up and retrying")
                    old = SharedMemory(name=self.name)
                    old.close()
                    old.unlink()
                    self.shm = SharedMemory(create=True, size=self._size, name=self.name)
                    log.debug(f"Created new shared memory '{self.name}' with size {self._size}")
            else:
                self.shm = SharedMemory(name=self.name)
        except Exception as e:
            log.error(f"Failed to create/attach shared memory '{self.name}': {e}")
            raise
        return self.shm

    def cleanup(self):
        """Clean up the shared memory segment."""
        if self.shm is not None:
            try:
                self.shm.close()
                if self._owner:
                    self.shm.unlink()
                    log.debug(f"Unlinked shared memory '{self.name}'")
            except Exception as e:
                log.warning(f"Error during shared memory cleanup for '{self.name}': {e}")
            finally:
                self.shm = None
    