from dataclasses import dataclass
from typing import List, Optional
from multiprocessing.synchronize import Event as MPEvent
from multiprocessing.sharedctypes import Synchronized

@dataclass(frozen=True)
class SyncEvents:
    """
    Container for multiprocessing synchronization primitives used in experiment.
    """

    hit: MPEvent
    """Set when a hit event occurs."""

    action_end: MPEvent
    """Set when the current action block ends."""

    end: MPEvent
    """Set when the entire experiment should terminate."""

    new_tr: MPEvent
    """Set when a new TR is received."""

    shm_ready: MPEvent
    """Set when shared memory is ready for access."""

    server_ready: MPEvent
    """Set when the server process is ready."""

    tr_index: Synchronized
    """Shared integer tracking the current TR index."""


@dataclass(frozen=True)
class ActionState:
    """
    Represents the current state of an action block during the experiment.
    """

    action_onsets: List[int]
    """TR indices at which action blocks start."""

    action_offsets: List[int]
    """TR indices at which action blocks end."""

    in_action: bool
    """Whether the experiment is currently within an action block."""

    in_cooldown: bool
    """Whether the experiment is currently within a cooldown period."""

    cooldown_end: Optional[int]
    """TR index at which the cooldown period ends, or ``None`` if no cooldown has occurred."""

    hit: bool
    """Whether a hit occurred."""
