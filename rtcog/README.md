# rtcog package

This directory contains the installable Python package and its command-line
entry points. `main.py` starts the full experiment application,
`run_minimal.py` starts the minimal workflow, and `paths.py` centralizes
project paths.

The subpackages separate scanner communication, experiment control, participant
interfaces, preprocessing, brain-state matching, processing orchestration,
visualization, configuration, resources, and shared utilities. User, operator,
extension, and API documentation is maintained in
[`docs/source`](../docs/source/index.rst) and published on
[Read the Docs](https://rtcog.readthedocs.io).
