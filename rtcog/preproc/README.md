# Real-time preprocessing

This package defines the configurable preprocessing pipeline applied to each
incoming fMRI volume. `pipeline.py` manages execution, `preproc_steps.py`
contains built-in steps, `step_types.py` names the supported stages, and
`custom_steps.py` is the extension point for project-specific processing.
