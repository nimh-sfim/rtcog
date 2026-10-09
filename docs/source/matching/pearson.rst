################
Pearson matching
################

Pearson matching reports the spatial correlation between each template map and
the current processed volume. Scores range from ``-1`` to ``1``. A positive
score indicates a similar spatial pattern, while a negative score indicates an
inverted pattern.

See :doc:`/matching` for the inputs and settings shared by all matching
methods.

Workflow summary
================

1. Prepare the masked templates, optionally scoring a representative processed
   run in the same command.
2. Review the generated files and any score traces.
3. Use the prepared ``.npz`` file as ``match_path`` in an ESAM run.

1. Prepare and evaluate offline
================================

The output directory must already exist. ``--data`` and
``--template_labels_path`` are optional. When ``--data`` is supplied,
``--discard`` leaves that many initial volumes unscored.

.. code:: bash

   python rtcog/matching/offline/pearson.py \
      --templates_path path/to/templates.nii \
      --mask path/to/mask.nii \
      --template_labels_path path/to/template_labels.txt \
      --data path/to/training_run.pp_Final.nii \
      --discard 100 \
      --out_dir ./existing_output_directory \
      --prefix pearson_method

Omit ``--data`` when only the runtime template file is needed. The optional
processed run should use the same mask and preprocessing planned for the ESAM
run.

2. Review the offline outputs
=============================

The command always writes:

``pearson_method.pearson_templates.npz``
   Template labels and masked template values. Use this file as the online
   ``match_path``.

When ``--data`` is supplied, it also writes:

``pearson_method.pearson_scores.npy``
   Score array with shape ``(n_templates, n_timepoints)``. Discarded volumes
   contain zeros.

``pearson_method.pearson_score_traces.npz``
   The same scores stored as one named array per template label.

``pearson_method.pearson_scores.png``
   Quick-look plot of the score traces, useful when choosing ``hit_thr``.

Constant templates, constant volumes, and non-finite correlations receive a
score of zero.

3. Configure the online run
===========================

Add the method to the complete ESAM configuration:

.. code:: yaml

   match_path: path/to/pearson_method.pearson_templates.npz
   hit_thr: your_threshold

   matching:
     match_method: pearson
     match_start: 100
     vols_noaction: 45

The online analysis mask must preserve the voxel count and ordering used during
offline preparation. The real-time outputs are described in
:ref:`output-files`.
