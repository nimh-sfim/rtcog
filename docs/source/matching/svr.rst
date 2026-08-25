############
SVR matching
############

SVR matching uses one pretrained linear support vector regression model per
template. The models are trained offline from a previously processed 4D run and
then loaded by ``rtcog`` during the real-time run.

See :doc:`/matching` for the settings shared by all matching methods, including
``discard``, ``match_start``, ``vols_noaction``, and hit-threshold selection.

Workflow summary
================

1. Train the SVR models offline and create ``training_svr.pkl``.
2. Review the offline training summaries and evaluate the scores on
   representative data.
3. Set ``match_method: svr`` and use ``training_svr.pkl`` as ``match_path`` for
   the real-time run.

1. Prepare the input offline
============================

The offline command requires processed training data, template maps, a template
label file, and the analysis mask. The label file must contain one
comma-separated list in template-volume order. See
:ref:`template-label-file` for an example.

The output directory must already exist. The value passed to ``--discard`` is
the number of initial training volumes excluded from model fitting; ``100`` is
only an example.

.. code:: bash

   python rtcog/matching/offline/svr.py \
      --data path/to/training_data.nii \
      --templates_path path/to/templates.nii \
      --template_labels_path path/to/template_labels.txt \
      --mask path/to/mask.nii \
      --discard 100 \
      --out_dir ./existing_output_directory \
      --prefix training_svr

By default, Lasso regression generates the per-template training targets. Pass
``--no_lasso`` to use ordinary linear regression instead.

2. Review the offline results
=============================

The command writes:

``training_svr.pkl``
   Dictionary of trained models. This is the file used as the online
   ``match_path``.

``training_svr_training_vols.csv``
   Zero-based volume indices used for SVR training after the offline
   ``--discard`` value.

``training_svr_lm_R2.csv`` and ``training_svr_lm_z_labels.csv``
   Per-volume regression fit and normalized training targets.

``training_svr.png`` and ``training_svr.html``
   Static and interactive training summaries.

Review the summaries to confirm that the training data, labels, and template
ordering are correct. Evaluate the trained models on representative processed
data before choosing ``hit_thr``; SVR scores do not have a universal threshold.

3. Configure the online run
===========================

Add the following values to the complete ESAM run configuration. All displayed
numbers are examples and may be changed:

.. code:: yaml

   discard: 10
   match_path: path/to/training_svr.pkl
   hit_thr: your_threshold

   matching:
     match_method: svr
     match_start: 100
     vols_noaction: 45

The mask used for the online run must preserve the voxel count and ordering used
to prepare the SVR models. Start the run using the complete command described in
:doc:`/usage`.

4. Check the online results
===========================

The real-time run writes the common ESAM score, hit, action, and report files
described in :ref:`output-files`. If matching never begins, confirm that
``match_start`` is smaller than the total number of volumes and is greater than
or equal to the online ``discard`` value.
