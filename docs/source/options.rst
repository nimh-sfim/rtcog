.. _configuration-reference:

################
Option reference
################

All normal run settings may be placed in YAML. The CLI spellings below are
optional overrides you may pass when running ``rtcog``.

CLI-only options
================

``-c PATH``, ``--config PATH``
   Required path to the YAML configuration.

``-h``, ``--help``
   Display command-line help.

Required options
==========================

``exp_type`` (CLI: ``-e``, ``--exp_type``)
   Experiment type: ``basic``, ``esam``, or a registered custom type.

``mask_path`` (CLI: ``-m``, ``--mask_path``, ``--mask``)
   NIfTI mask defining the streamed voxel locations. Its nonzero voxel count
   must match the AFNI stream.

``nvols`` (CLI: ``--nvols``)
   Expected number of volumes. Receiving more volumes raises an overflow error.

``out_dir`` (CLI: ``--out_dir``)
   Existing output directory.

``out_prefix`` (CLI: ``--out_prefix``)
   Prefix used for output filenames.

General options
===============

``debug`` (CLI: ``-d``, ``--debug``)
   Enable debug logging.

``silent`` (CLI: ``-s``, ``--silent``)
   Restrict console logging to critical messages.

``tcp_port`` (CLI: ``-p``, ``--tcp_port``)
   AFNI receiver port. It must match ``AFNI_REALTIME_MP_HOST_PORT``.

``show_data`` (CLI: ``-S``, ``--show_data``)
   Print received AFNI data for low-level diagnostics.

``discard`` (CLI: ``--discard``)
   Number of initial volumes stored but excluded from preprocessing.

Output options
==============

``save_orig`` (CLI: ``--save_orig``)
   Save the original incoming masked time series.

``auto_save`` (CLI: ``--auto_save``)
   Attempt to finalize and save outputs after a processing error. Default:
   ``false``.

``snapshot`` (CLI: ``--snapshot``)
   Save ``new_snapshots.npz`` at the end of the run for testing purposes.

``snapshot_dir`` (CLI: ``--snapshot_dir``)
   Snapshot output directory. The repository's ``Simulation/outputs`` directory
   is used when this is omitted.

Experiment and GUI options
==========================

``no_action`` (CLI: ``--no_action``)
   Disable the experiment's registered ActionSeries. In ESAM this suppresses
   the participant questionnaire and operator stream.

``fullscreen`` (CLI: ``--fullscreen``, ``--fscreen``)
   Start the experiment interface in full-screen mode.

``q_path`` (CLI: ``--q_path``)
   ESAM question JSON path.

``test_latency`` (CLI: ``--test_latency``, ``--latency``)
   Enable receiver and trigger latency diagnostics.

ESAM options
======================

``hit_thr`` (CLI: ``--hit_thr``)
   Score threshold used by hit detection. Required for ESAM runs.

``match_path`` (CLI: ``--match_path``)
   Model or template file. Required for the ``mask`` and ``svr`` matchers; see
   :doc:`matching` for method-specific inputs.

Preprocessing ``steps`` (YAML-only)
===================================

The ``steps`` list is executed in YAML order. Every entry accepts:

``name``
   Registered step name. Built-in names are case-insensitive.

``enabled``
   Include the step when ``true``. Default: ``false`` when omitted.

``save``
   Retain and write the step's complete time series. Default: ``false``.

``suffix``
   Optional custom output filename suffix.

Built-in step-specific options are:

``ema``
   ``alpha`` controls the exponential moving average. Default: ``0.98``.

``iglm``
   ``num_polorts`` controls polynomial regressors (default: ``2``), and
   ``iGLM_motion`` includes the six AFNI motion regressors (default: ``true``).

``kalman``
   ``n_cores`` controls worker-process count. Default: ``10``.

``smooth``
   ``fwhm`` controls spatial smoothing in millimeters. Default: ``4``.

``snorm``
   Spatially standardizes each volume and has no additional options.

``tnorm``
   ``nvols_to_compute`` controls the baseline length. Default: ``50``.

``windowing``
   ``win_length`` controls the temporal window length. Default: ``4``.

``matching`` section (YAML-only, ESAM)
======================================

``match_method``
   Matching method: ``svr``, ``mask``, ``pearson``, or ``nmi``.

``match_start``
   Zero-based volume index at which matching begins.

``vols_noaction``
   Cooldown, in volumes, after an action ends before another hit may trigger.

``hits`` section (YAML-only, ESAM)
==================================

``nconsec_vols``
   Consecutive above-threshold volumes required to register a hit.

``nonline``
   Maximum number of templates allowed above threshold simultaneously.

``do_mot``
   Enable motion rejection.

``mot_thr``
   Motion threshold. Required when ``do_mot`` is ``true``.
