from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class HitOpts:
    nconsec_vols: int
    hit_thr: float
    nonline: int
    do_mot: bool
    mot_thr: Optional[float] = field(default=None)

    def __post_init__(self):
        if self.do_mot and self.mot_thr is None:
            raise ValueError("mot_thr must be provided if do_mot is True.")
