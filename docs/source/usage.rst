######
Usage
######


1. Update config
=================

This package relies on options specified in a YAML file. Start from
``rtcog/config/default_config.yaml`` and customize it for your run. The order of
preprocessing steps is preserved and controls execution order. The reference
below describes every built-in field, its behavior, and required-value rules.

Key configuration sections:

- **General Options:** Debug levels, TCP ports, etc.
- **Preprocessing Steps:** List of steps in order with parameters (e.g., smoothing, normalization)
- **GUI:** Display and survey configurations
- **Testing options:** Enable snapshot and latency testing
- **Matching:** Template matching method and thresholds for ESAM mode
- **Hit Options:** Parameters for hit detection in ESAM mode

.. _configuration-reference:

Configuration reference
-----------------------

``rtcog`` reads its base configuration from the YAML file supplied with ``-c``
or ``--config``. Command-line values explicitly provided by the user then
replace corresponding top-level YAML values. Nested sections such as ``steps``,
``matching``, and ``hits`` are configured in YAML rather than merged from CLI
arguments.

Start from ``rtcog/config/default_config.yaml``. A small custom configuration
that omits keys used by the selected processor can fail at run time.

Where each option belongs
^^^^^^^^^^^^^^^^^^^^^^^^^

The short version is:

1. ``-c`` / ``--config`` is **CLI-only**. It tells ``rtcog`` which YAML file to
   load and therefore cannot be placed inside that file.
2. Every top-level option in the table below may be supplied **either in YAML or
   on the CLI**. If both are present, the explicitly supplied CLI value wins.
3. ``steps``, ``matching``, ``hits``, and all values nested inside them are
   **YAML-only**.
4. ``-h`` / ``--help`` is a CLI action rather than a configuration value.

For reproducible runs, the recommended approach is to keep run settings in YAML
and launch with only ``rtcog -c run_config.yaml``. Use CLI values for deliberate,
one-run overrides. The resolved values and launch command are both copied to the
output directory.

Top-level YAML and CLI names are not always identical:

.. list-table:: Top-level options accepted from YAML or CLI
   :header-rows: 1
   :widths: 24 28 48

   * - YAML key
     - CLI spelling
     - Notes
   * - ``debug``
     - ``-d``, ``--debug``
     - Boolean switch; use YAML to set ``false`` explicitly.
   * - ``silent``
     - ``-s``, ``--silent``
     - Boolean switch; use YAML to set ``false`` explicitly.
   * - ``tcp_port``
     - ``-p``, ``--tcp_port``
     - AFNI connection port.
   * - ``show_data``
     - ``-S``, ``--show_data``
     - Boolean switch.
   * - ``n_cores``
     - ``--ncores``
     - Accepted at the top level, but the built-in Kalman step currently uses
       the YAML-only ``steps[].n_cores`` value instead.
   * - ``mask_path``
     - ``-m``, ``--mask``
     - Required for every run.
   * - ``nvols``
     - ``--nvols``
     - Required for every run.
   * - ``discard``
     - ``--discard``
     - Initial volumes excluded from preprocessing.
   * - ``out_dir``
     - ``--out_dir``
     - Required; directory must already exist.
   * - ``out_prefix``
     - ``--out_prefix``
     - Required.
   * - ``snapshot_dir``
     - ``--snapshot_dir``
     - Snapshot output location.
   * - ``auto_save``
     - ``--auto_save``
     - Boolean switch.
   * - ``exp_type``
     - ``-e``, ``--exp_type``
     - Required; normally ``basic`` or ``esam``.
   * - ``no_action``
     - ``--no_action``
     - Boolean switch.
   * - ``fullscreen``
     - ``--fscreen``
     - Note the different CLI spelling.
   * - ``q_path``
     - ``--q_path``
     - ESAM question JSON path or resource name.
   * - ``match_path``
     - ``--match_path``
     - Required for built-in ESAM matchers.
   * - ``hit_thr``
     - ``--hit_thr``
     - Required for ESAM.
   * - ``snapshot``
     - ``--snapshot``
     - Boolean switch.
   * - ``test_latency``
     - ``--latency``
     - Note the different CLI spelling.

The YAML-only structure is:

.. code:: yaml

   steps:
     - name: smooth
       enabled: true
       save: false
       fwhm: 4.0

   matching:
     match_method: mask
     match_start: 100
     vols_noaction: 45

   hits:
     nconsec_vols: 2
     nonline: 1
     do_mot: true
     mot_thr: 0.2

CLI boolean flags only turn a value on; there are no corresponding ``--no-*``
flags to turn most of them off. Put ``false`` in YAML when that is the desired
value.

Required run values
^^^^^^^^^^^^^^^^^^^

Every run requires these values, either in YAML or on the command line:

``exp_type``
   ``basic`` or ``esam``. The full ``rtcog`` entry point also accepts a custom
   experiment name registered in ``rtcog/experiment_registry.py``. The minimal
   entry point only accepts ``basic`` and ``esam``.

``mask_path`` / ``--mask``
   NIfTI mask defining the voxel locations represented by the AFNI stream. The
   number of nonzero mask voxels must equal the number of incoming values.

``nvols``
   Expected number of volumes. Receiving more volumes raises an overflow error;
   zero volumes is rejected.

``out_dir``
   Existing output directory. ``rtcog`` writes the resolved configuration and
   launch command before starting its worker processes, so this directory must
   already exist.

``out_prefix``
   Prefix added to run output filenames.

ESAM additionally requires ``match_path``, ``hit_thr``, and the ``matching`` and
``hits`` YAML sections described below.

All of these values may be stored in YAML. For example, add the following
run-specific values to a copy of the shipped default configuration:

.. code:: yaml

   exp_type: basic
   mask_path: /absolute/path/to/mask.nii
   nvols: 300
   out_dir: /absolute/path/to/existing/output
   out_prefix: sub-001_run-01

Then the complete launch command is simply:

.. code:: bash

   rtcog -c path/to/run_config.yaml

For ESAM, also put ``match_path`` and ``hit_thr`` at the top level, set
``exp_type: esam``, and configure the ``matching`` and ``hits`` sections. A CLI
value overrides the same top-level YAML key, so ``--nvols 200`` would replace
``nvols: 300`` for that invocation.

General and saving options
^^^^^^^^^^^^^^^^^^^^^^^^^^

``debug``
   Enable debug-level logging. Default: ``false``.

``silent``
   Restrict logging to warnings and errors. Default: ``false``.

``tcp_port``
   Port on which ``rtcog`` listens for the AFNI real-time connection. Default:
   ``53214``. This must match ``AFNI_REALTIME_MP_HOST_PORT``.

``show_data``
   Print received AFNI data for low-level diagnostics. Default: ``false``.

``discard``
   Number of initial volumes stored but excluded from preprocessing. Default:
   ``10``.

``save_orig``
   Save the incoming masked data as ``<prefix>.orig.nii``. Default: ``false``.

``auto_save``
   Attempt to finalize and save processor outputs when the receiver encounters
   an error. Default: ``false``.

``snapshot`` and ``snapshot_dir``
   Save a ``new_snapshots.npz`` diagnostic snapshot. If ``snapshot_dir`` is
   omitted, the repository's ``Simulation/outputs`` path is used.

Preprocessing pipeline
^^^^^^^^^^^^^^^^^^^^^^

The ``steps`` list is executed in YAML order. Every entry accepts ``name``,
``enabled``, and ``save``. Names are case-insensitive. Setting ``save`` retains
and writes the step's complete time series, consuming an additional
``n_voxels * nvols`` array.

Built-in steps and their additional options are:

``ema``
   Exponential moving-average detrending. ``alpha`` defaults to ``0.98``.

``iglm``
   Incremental GLM nuisance regression. ``num_polorts`` defaults to ``2``;
   ``iGLM_motion`` defaults to ``true`` and includes the six AFNI motion
   parameters when enabled.

``kalman``
   Parallel Kalman filtering. ``n_cores`` defaults to ``10``. Set it inside the
   step entry; the top-level ``--ncores`` CLI option is not currently propagated
   into this nested configuration.

``smooth``
   Spatial Gaussian smoothing within the supplied mask. ``fwhm`` defaults to
   ``4``.

``snorm``
   Spatially standardize each processed volume across voxels.

``tnorm``
   Convert values to signal percent change after estimating a baseline.
   ``nvols_to_compute`` defaults to ``50`` processed volumes.

``windowing``
   Exponentially weighted temporal window. ``win_length`` defaults to ``4``.

Experiment and GUI options
^^^^^^^^^^^^^^^^^^^^^^^^^^

``no_action``
   Run the processor without its ActionSeries. In ESAM this suppresses the
   participant questionnaire and operator streaming process. Default: ``false``.

``fullscreen``
   Make full screen the default selection in the experiment setup dialog.
   Default: ``false``.

``q_path``
   ESAM question JSON. A full path is used directly; a basename such as
   ``questions_v1`` is resolved in ``rtcog/resources`` and receives a ``.json``
   extension automatically.

``test_latency``
   Enable receiver timing diagnostics. The ``--latency`` CLI flag sets this to
   ``true``.

Matching options
^^^^^^^^^^^^^^^^

The ESAM ``matching`` mapping has three required fields:

``match_method``
   One of ``svr``, ``mask``, ``pearson``, or ``nmi``. See :doc:`matching` for
   method-specific input files.

``match_start``
   Zero-based volume index at which matching begins. Volumes before this index
   are still preprocessed.

``vols_noaction``
   Minimum number of volumes after the previous action ends before another hit
   may trigger an action.

``match_path`` is a top-level option naming the model or template file, and
``hit_thr`` is the top-level score threshold used by hit detection. Both are
required for all built-in ESAM matchers.

Hit detection options
^^^^^^^^^^^^^^^^^^^^^

The ESAM ``hits`` mapping accepts:

``nconsec_vols``
   Number of consecutive volumes for which the selected template must meet or
   exceed ``hit_thr``.

``nonline``
   Maximum number of templates that may meet or exceed the threshold at once.
   If that count is exceeded, no hit is registered. Otherwise, the highest
   scoring qualifying template is selected.

``do_mot``
   Apply motion rejection when ``true``.

``mot_thr``
   Motion threshold used by the detector. It is required when ``do_mot`` is
   ``true``.

Shipped default
^^^^^^^^^^^^^^^

The maintained default configuration is included below for convenient review:

.. literalinclude:: ../../rtcog/config/default_config.yaml
   :language: yaml

2. Real-Time Scanner Setup
===========================

See :doc:`startup_afni`.

3. Running ``rtcog``
=====================

Basic Mode
----------

For basic preprocessing *without* template matching, the recommended YAML-driven
command is:

.. code:: bash

   conda activate rtcog
   rtcog -c path/to/basic_run.yaml

The following equivalent form demonstrates one-run CLI overrides. The YAML file
must still contain YAML-only settings such as ``steps``:

.. code:: bash

   conda activate rtcog

   rtcog \
     -c path/to/your_config.yaml \
     --exp_type basic \
     --nvols number_of_volumes \
     --mask path/to/your_mask.nii \
     --out_dir path/to/output_directory \
     --out_prefix your_output_prefix

ESAM Mode
---------

For preprocessing *with* template matching, the recommended YAML-driven command
is:

.. code:: bash

   conda activate rtcog
   rtcog -c path/to/esam_run.yaml

The YAML must contain ``steps``, ``matching``, and ``hits``. This equivalent form
demonstrates CLI overrides for the top-level run values:

.. code:: bash

   conda activate rtcog
 
   rtcog \
     -c path/to/your_config.yaml \
     --exp_type esam \
     --nvols number_of_volumes \
     --mask path/to/your_mask.nii \
     --out_dir path/to/output_directory \
     --out_prefix your_output_prefix \
     --hit_thr your_threshold \
     --match_path path/to/template_data.npz

Minimal Mode
------------

With conda
^^^^^^^^^^

If you installed Minimal mode to run without GUI dependencies, simply replace
``rtcog`` in the above commands with ``rtcog_min``:

.. code:: bash

   rtcog_min [options]

.. _docker-usage:

With Docker
^^^^^^^^^^^^

If you're using the Docker image instead, publish the scanner TCP port and mount a
local directory containing your config, mask, input data, and output location:

.. code:: bash

   docker run --rm --platform linux/amd64 \
     -p 53214:53214 \
     -v "$PWD:/work" \
     rtcog -c /work/path/to/your_config.yaml \
       --exp_type basic \
       --out_dir /work/path/to/output_directory \
       --nvols number_of_volumes \
       --out_prefix your_output_prefix \
       --mask /work/path/to/your_mask.nii


Any paths passed to ``rtcog_min`` or written inside the YAML config must be valid
inside the container. For example, a local file mounted with ``-v "$PWD:/work"``
should be referenced as ``/work/<filename>`` from inside the container.

The output directory must already exist on the host before the command starts.
The full and minimal native entry points have the same requirement.

Tip: snapshot testing normally writes to the repository's configured
``Simulation/outputs`` location. If you are generating snapshots from Docker,
you can direct them into a mounted path with ``--snapshot_dir``:

.. code:: bash

   --snapshot_dir /work/path/to/snapshot_outputs

Command Line Options
====================

General Options
---------------

- ``-c, --config``: YAML configuration file path
- ``-d, --debug``: Enable debug logging
- ``-s, --silent``: Minimal text output
- ``-p, --tcp_port``: TCP port for scanner connection (default: 53214)
- ``--show_data``: Display received data in terminal
- ``--ncores``: Set the top-level ``n_cores`` value. The built-in Kalman step
  currently reads ``steps[].n_cores`` from YAML instead.

Data Options
------------

- ``--nvols``: Number of volumes expected during the scan
- ``--mask``: Path to brain mask NIfTI file
- ``--discard``: Number of initial volumes to discard

Output Options
--------------

- ``--out_dir``: Output directory path
- ``--out_prefix``: Prefix for output files
- ``--snapshot_dir``: Directory for snapshot test outputs
- ``--auto_save``: Automatically save outputs when an error is encountered

Experiment Options
------------------

- ``--exp_type``: Experiment type ('basic' or 'esam', or custom type if configured. See :doc:`custom_plugin`)
- ``--no_action``: Run without the registered ActionSeries, GUI, or ESAM live streaming
- ``--fscreen``: Make full screen the default GUI selection
- ``--q_path``: Path or packaged resource name for ESAM question JSON
- ``--hit_thr``: Hit detection threshold (ESAM mode)
- ``--match_path``: Path to matching templates/models (ESAM mode)

Testing Options
---------------

- ``--snapshot``: Save a pipeline-state snapshot at the end of the run
- ``--latency``: Enable receiver and trigger latency diagnostics

.. _output-files:

Outputs
=======

Output names below use ``<prefix>`` for ``out_prefix``. Unless another location
is stated, files are written under ``out_dir``.

Every completed Basic or ESAM run
---------------------------------

``<prefix>_Options.yaml``
   Resolved configuration after applying CLI overrides.

``<prefix>_command.txt``
   Shell-quoted command used to start the run.

``<prefix>.Motion.1D``
   Six AFNI motion estimates per received volume, tab-delimited.

``<prefix>.pp_Final.nii``
   Final preprocessed time series reconstructed into the supplied mask space.

Optional preprocessing outputs
------------------------------

``<prefix>.orig.nii``
   Original incoming masked time series when ``save_orig: true``.

When a preprocessing step has ``save: true``, its full output is retained and
written at the end of the run:

- ``<prefix>.pp_EMA.nii``
- ``<prefix>.pp_iGLM.nii`` and one
  ``<prefix>.pp_iGLM_<regressor>.nii`` file per nuisance regressor
- ``<prefix>.pp_Kalman_LPfilter.nii``
- ``<prefix>.pp_Smooth.nii``
- ``<prefix>.pp_Zscore.nii`` for ``snorm``
- ``<prefix>.pp_Tnorm.nii``
- ``<prefix>.pp_Windowed.nii``

With ``snapshot: true``, ``new_snapshots.npz`` is written to ``snapshot_dir``.
This filename does not include the run prefix, so use separate snapshot
directories when retaining multiple runs.

ESAM outputs
------------

``<prefix>.<match_method>_scores.npy``
   Match score array with shape ``(n_templates, nvols)``. For example, mask
   matching writes ``<prefix>.mask_scores.npy``.

``<prefix>.hits.npy``
   Binary hit array with shape ``(n_templates, nvols)``.

``<prefix>.action_onsets.txt`` and ``<prefix>.action_offsets.txt``
   Zero-based volume indices for action starts and ends, one index per line.

``<prefix>.Hit_<template>_<NN>.nii``
   Mean processed map contributing to each detected hit.

``<prefix>.dyn_report.html``
   Final matching-score report. It is produced when the full ESAM operator
   stream runs and by headless ``rtcog_min`` ESAM runs.

When the full ESAM ActionSeries is enabled, each completed action can also
produce:

``<prefix>.hit<NNN>.wav``
   Participant audio recording.

``<prefix>.<YYYYMMDD-HHMMSS>.LikertResponses<NNN>.txt``
   CSV-formatted question, rating, and response-time data despite the ``.txt``
   extension.

Diagnostic outputs and logs
---------------------------

Latency mode writes receiver timing data and, when the full GUI is active,
trigger timing data and a latency plot. These diagnostics are intended for
hardware validation rather than normal analysis outputs.

``main.log`` is created in the directory from which ``rtcog`` is launched, not
in ``out_dir``. It is opened in write mode, so starting another run from the same
working directory replaces the previous log. Preserve or rename it between runs
when the log is part of the experiment record.
