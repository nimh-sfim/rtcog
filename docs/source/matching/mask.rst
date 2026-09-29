#############
Mask matching
#############

Mask matching measures average activity within weighted template masks. The
template masks are prepared before the real-time run. Processed run data are
optional and are only needed to generate offline activity traces.

See :doc:`/matching` for the settings shared by all matching methods, including
``discard``, ``match_start``, ``vols_noaction``, and hit-threshold selection.

Workflow summary
================

1. Prepare the weighted masks and create ``mask_method.template_data.npz``.
2. Optionally score a processed training run and review its activity traces.
3. Set ``match_method: mask`` and use the template-data file as ``match_path``
   for the real-time run.

1. Prepare the input offline
============================

The offline command requires template maps, a template label file, and the
analysis mask. The comma-separated labels must follow template-volume order.
See :ref:`template-label-file` for an example.

The output directory must already exist. In the example below, ``--thr 10`` is
a configurable example value. Set it for your template's value scale.

.. code:: bash

   python rtcog/matching/offline/mask.py \
      --templates_path path/to/templates.nii \
      --template_labels_path path/to/template_labels.txt \
      --mask path/to/mask.nii \
      --template_type continuous \
      --thr 10 \
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

The template-only command writes:

``mask_method.template_data.npz``
   Labels, voxel-selection masks, masked template weights, and voxel counts.
   This is the file used as the online ``match_path``.

2. Optionally score a processed run
===================================

To generate offline activity traces, first run ``rtcog`` in Basic mode using
the acquisition setup, analysis mask, and preprocessing planned for the ESAM
run. Then repeat the preparation command with the resulting
``<prefix>.pp_Final.nii`` file supplied as ``--data`` and set ``--discard`` to
the number of initial volumes to exclude from offline scoring.

When ``--data`` is provided, the command additionally writes:

``mask_method.act_traces.npz``
   Label-keyed offline activity traces, including zeros for discarded volumes.

``mask_method.traces.png`` and ``mask_method.traces.html``
   Static and interactive activity summaries.

Inspect the activity traces to decide on an appropriate
``hit_thr``. Mask scores depend on the template weights, selected voxels, and
data scale, so there is no universal threshold.

3. Configure the online run
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

4. Check the online results
===========================

The real-time run writes the common ESAM score, hit, action, and report files
described in :ref:`output-files`. If the scores do not resemble the offline
traces, check the analysis mask, voxel ordering, preprocessing, and input data
scale before changing the threshold.
