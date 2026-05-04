######
Usage
######


1. Update config
=================

This package relies on options specified in a yaml file to run. We've
provided you with a default setup at
``rtcog/config/default_config.yaml`` which you can customize to your
liking. Please note that the order of preprocessing steps is preserved,
so any changes to that will affect the pipeline.

Key configuration sections:

- **General Options:** Debug levels, TCP ports, etc.
- **Preprocessing Steps:** List of steps in order with parameters (e.g., smoothing, normalization)
- **GUI:** Display and survey configurations
- **Testing options:** Enable snapshot and latency testing
- **Matching:** Template matching method and thresholds for ESAM mode
- **Hit Options:** Parameters for hit detection in ESAM mode

2. Real-Time Scanner Setup
===========================

See :doc:`startup_afni`.

3. Running ``rtcog``
=====================

Basic Mode
----------

For basic preprocessing *without* template matching:

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

For preprocessing *with* template matching:

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
- ``--hit_thr``: Hit detection threshold (ESAM mode)
- ``--match_path``: Path to matching templates/models (ESAM mode)

Outputs
=======

``rtcog`` generates several output files in the specified output directory:

- ``{prefix}_Options.yaml``: Copy of the configuration used
- ``{prefix}_match_scores.npy``: Template matching scores (ESAM mode)
- To be continued...

Log files are also created for debugging and monitoring.
