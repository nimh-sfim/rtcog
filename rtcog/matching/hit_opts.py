from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class HitOpts:
    """
    Configuration options controlling hit detection logic.
    """

    nconsec_vols: int
    """Number of consecutive volumes required to register a hit."""

    hit_thr: float
    """Threshold value that must be met or exceeded to count as a hit."""

    nonline: int
    """Maximum number of templates allowed to exceed the hit threshold
    simultaneously.

    If more than ``nonline`` templates meet the threshold at the same time,
    no hit is registered. For example, a value of ``2`` means at most two
    templates may meet the threshold for a hit to be counted; the template with
    the greatest value is selected as the hit.
    """

    do_mot: bool
    """Whether motion thresholding should be applied."""

    mot_thr: Optional[float] = field(default=None)
    """Motion threshold value. Required if ``do_mot`` is ``True``."""

    def __post_init__(self):
        if self.do_mot and self.mot_thr is None:
            raise ValueError("mot_thr must be provided if do_mot is True.")
