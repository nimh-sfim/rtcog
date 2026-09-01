#####
Usage
#####

Quickstart
==========

1. Copy the :download:`default configuration
   <../../rtcog/config/default_config.yaml>` and give the copy a name for the
   run:

   .. code-block:: bash

      cp rtcog/config/default_config.yaml run_config.yaml

2. Adjust the values as needed.

3. Start ``rtcog`` and wait for ``5) Ready to go...`` before beginning the
   acquisition:

   .. code-block:: bash

      conda activate rtcog
      rtcog \
        -c run_config.yaml \
        --exp_type basic \
        --nvols number_of_volumes \
        --mask /path/to/mask.nii \
        --out_dir /path/to/output_directory \
        --out_prefix output_prefix

   ``out_dir`` must already exist. The AFNI realtime mask and ``--mask`` file
   must contain the same voxels.

See :doc:`startup_afni` for the AFNI realtime settings and :doc:`simulation`
for a complete simulated run.

ESAM runs
---------

Configure the existing ``matching`` and ``hits`` sections in the YAML file,
then supply the experiment type and run-specific matching inputs on the CLI:

.. code-block:: bash

   rtcog -c run_config.yaml \
     --exp_type esam \
     --nvols number_of_volumes \
     --mask /path/to/mask.nii \
     --out_dir /path/to/output_directory \
     --out_prefix output_prefix \
     --hit_thr hit_threshold \
     --match_path /path/to/matching_input.npz

Omit ``--match_path`` when the selected matcher does not require an input
file. See :doc:`matching` for the accepted matcher inputs.

Minimal mode
------------

When GUI dependencies are not installed, use the same YAML and CLI options but
replace ``rtcog`` with ``rtcog_min``.

.. _docker-usage:

Docker
------

Paths inside the YAML file must use their locations inside the container. For
example, with the config, mask, and output directory beneath the current
directory:

.. code-block:: bash

   docker run --rm --platform linux/amd64 \
     -p 53214:53214 \
     -v "$PWD:/work" \
     rtcog \
       -c /work/run_config.yaml \
       --exp_type basic \
       --nvols number_of_volumes \
       --mask /work/path/to/mask.nii \
       --out_dir /work/path/to/output_directory \
       --out_prefix output_prefix

Use paths beginning with ``/work`` inside ``run_config.yaml``. Create the host
output directory before starting the container.

YAML versus CLI
===============

``rtcog`` reads its base configuration from the YAML file supplied with ``-c``
or ``--config``. Command-line values explicitly provided by the user then
override corresponding top-level YAML values. Nested sections such as ``steps``,
``matching``, and ``hits`` must be configured in YAML rather than supplied via
CLI.

You can define all parameters in the YAML file if you wish. However, it is often
easiest to pass run-specific values via CLI at runtime because they often change.

These values include:

- ``exp_type``, ``nvols``, ``mask``, ``out_dir``, and ``out_prefix``.
- ESAM runs also pass ``hit_thr`` and, when required, ``match_path``.

When a value is provided both in the YAML and CLI, the CLI value wins.
The final configuration used is saved to ``<out_prefix>_Options.yaml``
in ``out_dir`` for reproducibility.

See the :doc:`complete option reference <options>` for every YAML setting and
its corresponding CLI override.

.. _output-files:

Outputs
========

Unless stated otherwise, outputs are written in ``out_dir`` and use
``out_prefix`` in place of ``<prefix>``.

Every completed run
-------------------

``<prefix>_Options.yaml``
   Resolved YAML and CLI configuration used for the run.

``<prefix>.Motion.1D``
   Six AFNI motion estimates per received volume.

``<prefix>.pp_Final.nii``
   Final preprocessed time series reconstructed into the supplied mask space.

Optional preprocessing outputs
------------------------------

``<prefix>.orig.nii`` is written when ``save_orig: true``. A preprocessing
step with ``save: true`` writes its full time series using the corresponding
suffix:

- ``<prefix>.pp_EMA.nii``
- ``<prefix>.pp_iGLM.nii`` and ``<prefix>.pp_iGLM_<regressor>.nii``
- ``<prefix>.pp_Kalman_LPfilter.nii``
- ``<prefix>.pp_Smooth.nii``
- ``<prefix>.pp_Zscore.nii``
- ``<prefix>.pp_Tnorm.nii``
- ``<prefix>.pp_Windowed.nii``

With ``snapshot: true``, ``new_snapshots.npz`` is written to ``snapshot_dir``.
The filename does not contain the run prefix.

ESAM outputs
------------

``<prefix>.<match_method>_scores.npy``
   Match scores with shape ``(n_templates, nvols)``.

``<prefix>.hits.npy``
   Binary hit array with shape ``(n_templates, nvols)``.

``<prefix>.action_onsets.txt`` and ``<prefix>.action_offsets.txt``
   Zero-based volume indices for action starts and ends.

``<prefix>.Hit_<template>_<NN>.nii``
   Mean processed map contributing to each detected hit.

``<prefix>.dyn_report.html``
   Final matching-score report.

Full ESAM runs with actions enabled can additionally write participant audio as
``<prefix>.hit<NNN>.wav`` and questionnaire responses as
``<prefix>.<YYYYMMDD-HHMMSS>.LikertResponses<NNN>.txt``.

Diagnostics and logs
--------------------

Latency mode writes receiver timing data and, with the full GUI, trigger timing
data and a latency plot. ``main.log`` is written in the directory from which
``rtcog`` is launched, not in ``out_dir``. Starting another run from that
directory replaces the log.
