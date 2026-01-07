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

``rtcog`` integrates with AFNI's real-time fMRI infrastructure.


You can start up afni in real-time mode by executing:

.. code:: bash

   export AFNI_REALTIME_Registration=3D:_realtime
   export AFNI_REALTIME_MP_HOST_PORT=localhost:53214
   export AFNI_REALTIME_SEND_VER=YES
   export AFNI_REALTIME_SHOW_TIMES=YES
   export AFNI_REALTIME_Function=FIM
   export AFNI_REALTIME_Graph=Realtime
   export AFNI_REALTIME_Mask_Vals=All_Data_light

   afni -rt

Detailed AFNI setup instructions to come...

1. Running ``rtcog``
=====================

Basic Mode
----------

For basic preprocessing *without* template matching:

.. code:: bash

   conda activate rtcog

   rtcog \
     -c path/to/your_config.yaml \            # Path to your YAML config
     --exp_type basic \                       # Experiment type
     --nvols 300 \                            # Number of volumes in your dataset
     --mask path/to/your_mask.nii \           # Your mask file
     --out_dir ./output_directory \           # Where results will be saved
     --out_prefix your_output_prefix \        # Prefix for output files

ESAM Mode
---------

For experiment control *with* template matching:

.. code:: bash

   conda activate rtcog

   rtcog \
     -c path/to/your_config.yaml \            # Path to your YAML config
     --exp_type esam \                        # Experiment type
     --nvols 300 \                            # Number of volumes in your scan
     --mask_path path/to/your_mask.nii \      # Your mask file
     --out_dir ./output_directory \           # Where results will be saved
     --out_prefix your_output_prefix \        # Prefix for output files
     --hit_thr your_threshold                 # Threshold for hit (float)
     --match_path path/to/template_data.npz \ # Template matching input

Minimal Mode
------------

For running without GUI dependencies (useful for testing purposes):

.. code:: bash

   python -m rtcog.run_minimal [options]

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
- ``--save_orig``: Save original preprocessed data

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
