from dataclasses import dataclass
from typing import List, Optional

@dataclass
class QAState:
    qa_onsets: List[int]
    qa_offsets: List[int]
    in_qa: bool
    in_cooldown: bool
    cooldown_end: Optional[int]
