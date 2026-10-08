"""Regenerate the synthetic NIfTI inputs used by the snapshot test.

The generated volumes are a mathematical phantom. They do not contain human
participant or scanner data. Run this script from any directory after changing
the phantom definition intentionally; normal snapshot-test runs use the
committed NIfTI files and do not invoke this script.

Each volume is evaluated analytically after a small rigid-body transform. This
avoids cumulative interpolation error while giving AFNI known motion to estimate.
"""

from pathlib import Path

import nibabel as nib
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
SHAPE = (16, 16, 12)
NVOLS = 24


def rigid_motion(t: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic translations in voxels and rotations in degrees."""
    phase = 2 * np.pi * t / NVOLS
    translation = np.array(
        [
            0.35 * np.sin(phase),
            0.28 * np.sin(2 * phase),
            0.22 * np.sin(3 * phase),
        ]
    )
    rotation = np.array(
        [
            0.45 * np.sin(4 * phase),
            0.35 * np.sin(5 * phase),
            0.30 * np.sin(6 * phase),
        ]
    )
    return translation, rotation


def rotation_matrix(rotation_degrees: np.ndarray) -> np.ndarray:
    """Construct an x-y-z rigid rotation matrix."""
    rx, ry, rz = np.deg2rad(rotation_degrees)
    cx, cy, cz = np.cos((rx, ry, rz))
    sx, sy, sz = np.sin((rx, ry, rz))

    rotate_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rotate_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotate_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rotate_z @ rotate_y @ rotate_x


def main() -> None:
    x, y, z = np.indices(SHAPE, dtype=np.float64)
    center = (np.asarray(SHAPE, dtype=np.float64) - 1) / 2
    radius = np.sqrt(
        (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    )

    mask = (radius <= 5).astype(np.uint8)
    centered = np.stack(
        (x - center[0], y - center[1], z - center[2]), axis=0
    )

    data = np.empty((*SHAPE, NVOLS), dtype=np.int16)
    for t in range(NVOLS):
        translation, rotation = rigid_motion(t)
        sample_at = rotation_matrix(rotation).T @ (
            centered.reshape(3, -1) - translation[:, np.newaxis]
        )
        sample_x, sample_y, sample_z = sample_at.reshape((3, *SHAPE))

        ellipsoid = sample_x**2 / 18 + sample_y**2 / 13 + sample_z**2 / 9
        asymmetric_feature = 180 * np.exp(
            -(
                (sample_x - 2.4) ** 2 / 2.5
                + (sample_y + 1.6) ** 2 / 1.8
                + (sample_z - 0.8) ** 2 / 1.2
            )
        )
        phantom = 900 + 500 * np.exp(-ellipsoid) + asymmetric_feature
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
