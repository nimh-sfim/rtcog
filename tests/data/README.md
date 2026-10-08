# Test data

This directory contains compact synthetic fixtures used by unit, integration,
and snapshot tests. `generate_snapshot_input.py` deterministically creates the
synthetic NIfTI input and mask. The phantom has nonconstant voxel intensities,
an asymmetric spatial feature, temporal intensity changes, and small
deterministic translations and rotations. Those properties exercise masking,
unmasking, smoothing, spatial normalization, and AFNI motion estimation without
requiring local participant or scanner data.

Regenerate the expected preprocessing snapshot with the three-terminal AFNI
workflow in `docs/source/testing.rst`; the input generator does not create that
baseline.
