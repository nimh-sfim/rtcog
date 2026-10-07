# Test data

This directory contains compact synthetic fixtures used by snapshot and
integration-style tests. `generate_snapshot_input.py` deterministically creates
the synthetic NIfTI input, mask, and expected preprocessing snapshot.

Only small, reproducible fixtures belong here; full scan data and local run
outputs should remain outside version control.
