from abc import ABC, abstractmethod
from typing import List

from rtfmri.viz.plotter import Plotter

class BaseStreamer(ABC):
    """
    Abstract base class for realtime fMRI streamers.

    This class defines the interface and common behavior expected from any
    streamer that handles real-time visualization of fMRI data or similar.

    Subclasses must implement the `run` and `update` methods.

    Attributes
    ----------
    _plotters : List[Plotter]
        A list of plotter objects responsible for visualizing different types of data.
    """
    def __init__(self):
        """
        Initialize the base streamer.

        This sets up the internal list of plotters. Subclasses should append
        their plotters to this list so that they can be cleanly shut down.
        """
        self._plotters: List[Plotter] = []

    @abstractmethod
    def run(self):
        """Launch the streaming loop and server.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def update(self):
        """
        Perform one step of the streaming update loop.

        This method should wait for new data (e.g., a new TR), check if any
        visual updates are needed, and update the plotters accordingly.
        Must be implemented by subclasses.
        """
        pass

    def _shutdown(self):
        """
        Clean up resources when streaming ends.

        This includes closing any open plotters and releasing other shared
        resources. Subclasses may override this to add additional cleanup logic.
        """
        for plotter in self._plotters:
            plotter.close()
