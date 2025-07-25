from dataclasses import dataclass
from typing import List, Optional
from multiprocessing.synchronize import Event as MPEvent
from multiprocessing.sharedctypes import Synchronized

@dataclass(frozen=True)
class SyncEvents:
    hit: MPEvent
    qa_end: MPEvent
    end: MPEvent
    new_tr: MPEvent
    shm_ready: MPEvent
    tr_index: Synchronized

@dataclass(frozen=True)
class QAState:
    qa_onsets: List[int]
    qa_offsets: List[int]
    in_qa: bool
    in_cooldown: bool
    cooldown_end: Optional[int]
    hit: bool
