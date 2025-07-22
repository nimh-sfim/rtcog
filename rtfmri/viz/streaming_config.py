from dataclasses import dataclass
from typing import List
from types import SimpleNamespace
from nibabel.nifti1 import Nifti1Image

@dataclass(frozen=True)
class StreamingConfig:
    Nt: int
    template_labels: List[str]
    hit_thr: float
    matching_opts: SimpleNamespace
    mask_img: Nifti1Image
    Nv: int
