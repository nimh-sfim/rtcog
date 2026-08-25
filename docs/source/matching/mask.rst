#############
Mask matching
#############

Mask matching measures average activity within weighted template masks. The
template masks and their offline activity traces are prepared before the
real-time run.

See :doc:`/matching` for the settings shared by all matching methods, including
``discard``, ``match_start``, ``vols_noaction``, and hit-threshold selection.

Workflow summary
================

1. Prepare the weighted masks and create ``mask_method.template_data.npz``.
2. Review the offline activity traces and choose an appropriate hit threshold.
3. Set ``match_method: mask`` and use the template-data file as ``match_path``
   for the real-time run.

1. Prepare the input offline
============================

The offline command requires processed training data, template maps, a template
label file, and the analysis mask. The comma-separated labels must follow
template-volume order. See :ref:`template-label-file` for an example.

The output directory must already exist. The value passed to ``--discard`` is
the number of initial training volumes excluded from offline scoring; ``100``
is only an example.

.. code:: bash

   python rtcog/matching/offline/mask.py \
      --data path/to/training_data.nii \
      --templates_path path/to/templates.nii \
      --template_labels_path path/to/template_labels.txt \
      --mask path/to/mask.nii \
      --template_type normal \
      --thr 10 \
      --discard 100 \
      --out_dir ./existing_output_directory \
      --prefix mask_method

Use ``--template_type normal`` for continuous maps; only template values above
``--thr`` are selected. Use ``--template_type binary`` to select the nonzero
voxels of binary templates; ``--thr`` is then ignored.

2. Review the offline results
=============================

The command writes:

``mask_method.template_data.npz``
   Labels, voxel-selection masks, masked template weights, and voxel counts.
   This is the file used as the online ``match_path``.

``mask_method.masked_templates.nii.gz``
   Input templates after applying the analysis mask.

``mask_method.act_traces.npz``
   Label-keyed offline activity traces, including zeros for discarded volumes.

``mask_method.traces.png`` and ``mask_method.traces.html``
   Static and interactive activity summaries.

``mask_method.template_pairwise_stats.csv`` and ``mask_method.trace_pairwise_stats.csv``
   Spatial template comparisons and temporal trace correlations.

Inspect the activity traces from representative processed data before choosing
``hit_thr``. Mask scores depend on the template weights, selected voxels, and
data scale, so there is no universal threshold.

3. Configure the online run
===========================

Add the following values to the complete ESAM run configuration. Use
``match_method: mask``—not ``mask_method``. All displayed numbers are examples
and may be changed:

.. code:: yaml

   discard: 10
   match_path: path/to/mask_method.template_data.npz
   hit_thr: your_threshold

   matching:
     match_method: mask
     match_start: 100
     vols_noaction: 45

The online analysis mask must preserve the voxel count and ordering used during
offline preparation. Start the run using the complete command described in
:doc:`/usage`.

4. Check the online results
===========================

The real-time run writes the common ESAM score, hit, action, and report files
described in :ref:`output-files`. If the scores do not resemble the offline
traces, check the analysis mask, voxel ordering, preprocessing, and input data
scale before changing the threshold.
