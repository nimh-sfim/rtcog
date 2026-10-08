"""Regenerate the synthetic NIfTI inputs used by the snapshot test.

The generated volumes are a mathematical phantom. They do not contain human
participant or scanner data. Run this script from any directory after changing
the phantom definition intentionally; normal snapshot-test runs use the
committed NIfTI files and do not invoke this script.
"""

from pathlib import Path

import nibabel as nib
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
SHAPE = (16, 16, 12)
NVOLS = 24


def main() -> None:
    x, y, z = np.indices(SHAPE, dtype=np.float64)
    center = (np.asarray(SHAPE, dtype=np.float64) - 1) / 2
    radius = np.sqrt(
        (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    )

    mask = (radius <= 5).astype(np.uint8)
    ellipsoid = (
        (x - center[0]) ** 2 / 18
        + (y - center[1]) ** 2 / 13
        + (z - center[2]) ** 2 / 9
    )
    phantom = 900 + 500 * np.exp(-ellipsoid)

    data = np.empty((*SHAPE, NVOLS), dtype=np.int16)
    for t in range(NVOLS):
        scale = 1 + 0.012 * np.sin(2 * np.pi * t / 8) + 0.0008 * t
        data[..., t] = np.rint(phantom * scale).astype(np.int16)

    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    affine[:3, 3] = -center * 3

    input_img = nib.Nifti1Image(data, affine)
    input_img.header.set_xyzt_units("mm", "sec")
    input_img.header.set_zooms((3.0, 3.0, 3.0, 2.0))
    input_img.to_filename(OUTPUT_DIR / "synthetic_snapshot_input.nii.gz")

    mask_img = nib.Nifti1Image(mask, affine)
    mask_img.header.set_xyzt_units("mm")
    mask_img.to_filename(OUTPUT_DIR / "synthetic_snapshot_mask.nii.gz")


if __name__ == "__main__":
    main()
