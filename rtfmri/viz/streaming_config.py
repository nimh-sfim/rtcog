from dataclasses import dataclass
from typing import List, Optional
from types import SimpleNamespace
from multiprocessing.synchronize import Event as MPEvent
from nibabel.nifti1 import Nifti1Image

@dataclass(frozen=True)
class StreamerConfig:
    Nt: int
    template_labels: List[str]
    hit_thr: float
    matching_opts: SimpleNamespace
    mask_img: Nifti1Image
    Nv: int

@dataclass(frozen=True)
class SyncEvents:
    new_tr: MPEvent
    shm_ready: MPEvent
    qa_end: MPEvent
    hit: MPEvent
    end: MPEvent

@dataclass(frozen=True)
class QAState:
    qa_onsets: List[int]
    qa_offsets: List[int]
    in_qa: bool
    in_cooldown: bool
    cooldown_end: Optional[int]
