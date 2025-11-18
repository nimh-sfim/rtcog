######
Usage
######


1. Update config
=================

This package relies on options specified in a yaml file to run. We’ve
provided you with a default setup at
``rtcog/config/default_config.yaml`` which you can customize to your
setup. Please note that the order of preprocessing steps is preserved,
so any changes to that will affect the pipeline.

2. Real-Time Scanner Setup
===========================

work in progress...


3. Running ``rtcog``
=====================

In Preproc mode:

.. code:: bash

   conda activate rtcog

   rtcog \
     -c path/to/your_config.yaml \            # Path to your YAML config
     --nvols 300 \                            # Number of volumes in your dataset
     --mask path/to/your_mask.nii \      # Your mask file
     --out_dir ./output_directory \           # Where results will be saved
     --out_prefix your_output_prefix \        # Prefix for output files
     --exp_type preproc \                     # Experiment type

In ESAM mode:

.. code:: bash

   conda activate rtcog

   rtcog \
     -c path/to/your_config.yaml \            # Path to your YAML config
     --nvols 300 \                            # Number of volumes in your scan
     --mask_path path/to/your_mask.nii \      # Your mask file
     --out_dir ./output_directory \           # Where results will be saved
     --out_prefix your_output_prefix \        # Prefix for output files
     --exp_type esam \                        # Experiment type
     --hit_thr your_threshold                 # Threshold for hit (float)
     --match_path path/to/template_data.npz \ # Template matching input
    
Outputs
=======

work in progress...
