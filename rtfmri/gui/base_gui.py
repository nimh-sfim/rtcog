from abc import ABC, abstractmethod
from psychopy.visual import Window

class BaseGUI(ABC):
    registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Skip abstract or helper base classes
        if cls.__name__ == "BaseGUI" or cls.__name__.startswith("_"):
            return

        BaseGUI.registry[cls.__name__] = cls

    def _create_experiment_window(self) -> Window:
        """
        Create and return the PsychoPy experiment window.

        Returns
        -------
        psychopy.visual.Window
            The experiment display window.
        """
        return Window(
            fullscr=self.fscreen, screen=self.screen, size=(1920,1080),
            winType='pyglet', allowGUI=True, allowStencil=False,
            color=[0,0,0], colorSpace='rgb', blendMode='avg',
            useFBO=True, units='norm'
        )
    
    def _draw_stims(self, stims, flip=True) -> None:
        """
        Draw and optionally flip a list of stimuli.

        Parameters
        ----------
        stims : list
            List of PsychoPy stimulus objects to draw.
        flip : bool
            Whether to flip the window after drawing (default: True).
        """
        for stim in stims:
            stim.draw()
        if flip:
            self.ewin.flip()

    @abstractmethod
    def draw_resting_screen(self) -> None:
        """Draw the default resting screen (e.g., crosshair + text)."""
        pass

    @abstractmethod
    def run_full_QA(self) -> dict:
        """Run a full QA interaction and return responses."""
        pass

    @abstractmethod
    def poll_trigger(self) -> None:
        """Poll for triggers or escape key (for latency testing)."""
        pass

    @abstractmethod
    def close_psychopy_infrastructure(self) -> None:
        """Close any GUI windows and clean up PsychoPy resources."""
        pass

    @abstractmethod
    def save_trigger(self) -> None:
        """Save collected trigger timings to disk."""
        pass
    