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
    action_end: MPEvent
    end: MPEvent
    new_tr: MPEvent
    shm_ready: MPEvent
    server_ready: MPEvent
    tr_index: Synchronized

@dataclass(frozen=True)
class ActionState:
    """
    Represents the current state of an action block during the experiment.
    """
    action_onsets: List[int]
    action_offsets: List[int]
    in_action: bool
    in_cooldown: bool
    cooldown_end: Optional[int]
    hit: bool
