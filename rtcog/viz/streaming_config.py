from dataclasses import dataclass
from typing import List
from nibabel.nifti1 import Nifti1Image

from rtfmri.matching.matching_opts import MatchingOpts

@dataclass(frozen=True)
class StreamingConfig:
    """
    Configuration container for real-time fMRI data streaming and matching.
    """
    Nt: int
    template_labels: List[str]
    hit_thr: float
    matching_opts: MatchingOpts
    mask_img: Nifti1Image
    Nv: int
    out_dir: str
    out_prefix: str
