# Test data

This directory contains compact synthetic fixtures used by snapshot and
integration-style tests. `generate_snapshot_input.py` deterministically creates
the synthetic NIfTI input and mask.

Regenerate the expected preprocessing snapshot with the three-terminal AFNI
workflow in `docs/source/testing.rst`; the input generator does not create that
baseline.
