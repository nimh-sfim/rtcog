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

1. Run ``rtcog`` in Basic mode on a training rest run.
2. Pass the processed training run to the offline command to prepare the weighted
   masks and create ``mask_method.template_data.npz``.
3. Review the offline activity traces and choose an appropriate hit threshold.
4. Set ``match_method: mask`` and use the template-data file as ``match_path``
   for the real-time run.

1. Process a training run
=========================

First, run ``rtcog`` in Basic mode on a training rest run. Use the same
acquisition setup, analysis mask, and preprocessing steps planned for the later
ESAM run. The resulting ``<prefix>.pp_Final.nii`` file is the processed training
run used by the offline mask command. See :ref:`output-files` for details about
this output.

2. Prepare the input offline
============================

The offline command requires the ``rtcog``-processed training run, template maps,
a template label file, and the analysis mask. Pass the
``<prefix>.pp_Final.nii`` file from the Basic-mode training run to ``--data``.
The comma-separated labels must follow template-volume order. See
:ref:`template-label-file` for an example.

The output directory must already exist. In the example below, ``--thr 10`` and
``--discard 100`` are configurable example values. Set ``--thr`` for your
template's value scale and ``--discard`` to the number of initial training
volumes to exclude from offline scoring.

.. code:: bash

   python rtcog/matching/offline/mask.py \
      --data path/to/training_run.pp_Final.nii \
      --templates_path path/to/templates.nii \
      --template_labels_path path/to/template_labels.txt \
      --mask path/to/mask.nii \
      --template_type continuous \
      --thr 10 \
      --discard 100 \
      --out_dir ./existing_output_directory \
      --prefix mask_method

``--template_type`` controls how voxels are selected from each template after
the analysis mask is applied:

``continuous``
   Use for full statistical or weighted template maps. Values above ``--thr``
   are selected and retained as weights. Choose a threshold in the template's
   value scale.

``binary``
   Use when selected clusters or regions have already been saved as a ``0``/``1``
   mask. Every nonzero voxel is selected, ``--thr`` is ignored, and the original
   nonzero values are retained as weights. A ``0``/``1`` mask therefore gives
   every selected voxel equal weight. This avoids re-thresholding clusters
   extracted from a full template.

3. Review the offline results
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

Inspect the activity traces to decide on an appropriate
``hit_thr``. Mask scores depend on the template weights, selected voxels, and
data scale, so there is no universal threshold.

4. Configure the online run
===========================

Add the following values to the complete ESAM run configuration. 
All displayed numbers are examples
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

5. Check the online results
===========================

The real-time run writes the common ESAM score, hit, action, and report files
described in :ref:`output-files`. If the scores do not resemble the offline
traces, check the analysis mask, voxel ordering, preprocessing, and input data
scale before changing the threshold.
