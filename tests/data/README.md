# Test data

This directory contains compact synthetic fixtures used by snapshot and
integration-style tests. `generate_snapshot_input.py` deterministically creates
the synthetic NIfTI input and mask. The phantom has intensity variation, an
asymmetric spatial feature, and small deterministic translations and rotations
so AFNI can produce motion estimates for the iGLM snapshot coverage.

Regenerate the expected preprocessing snapshot with the three-terminal AFNI
workflow in `docs/source/testing.rst`; the input generator does not create that
baseline.
