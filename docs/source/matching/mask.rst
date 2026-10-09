#############
Mask matching
#############

Mask matching measures average activity within weighted template masks.

See :doc:`/matching` for the settings shared by all matching methods, including
``discard``, ``match_start``, ``vols_noaction``, and hit-threshold selection.

Workflow summary
================

1. Prepare the weighted masks, optionally scoring a processed training run in
   the same command.
2. Set ``match_method: mask`` and use the template-data file as ``match_path``
   for the real-time run.
3. Compare the online results with any offline activity traces.

1. Prepare the input and optionally score data
==============================================

The offline command requires template maps, a template label file, and the
analysis mask. The comma-separated labels must follow template-volume order.
See :ref:`template-label-file` for an example.

The output directory must already exist. In the example below, ``--thr 10`` is
a configurable example value, not a recommended default for every template.

.. code:: bash

   python rtcog/matching/offline/mask.py \
      --templates_path path/to/templates.nii \
      --template_labels_path path/to/template_labels.txt \
      --mask path/to/mask.nii \
      --template_type continuous \
      --thr 10 \
      --out_dir ./existing_output_directory \
      --prefix mask_method

How template thresholding works
-------------------------------

The analysis mask is applied first. ``--template_type`` then determines which
template voxels contribute to the score:

``continuous``
   Selects template values greater than ``--thr``. With a positive threshold,
   only positive template values above that threshold are selected. Their
   original values are retained as weights.

``binary``
   Selects every nonzero template voxel and ignores ``--thr``. Original values
   are retained, so a ``0``/``1`` mask gives all selected voxels equal weight.

Thresholding removes background and weaker template values and fixes the voxel
selection used throughout the run. It applies to the template, not the incoming
activity, which may be positive or negative.

Optional scoring during preparation
-----------------------------------

To generate offline activity traces, first run ``rtcog`` in Basic mode using
the acquisition setup, analysis mask, and preprocessing planned for the ESAM
run. Add the resulting ``<prefix>.pp_Final.nii`` file to the same preparation
command with ``--data``. Use ``--discard`` to leave initial volumes out of the
offline scoring:

.. code:: bash

   python rtcog/matching/offline/mask.py \
      --templates_path path/to/templates.nii \
      --template_labels_path path/to/template_labels.txt \
      --mask path/to/mask.nii \
      --template_type continuous \
      --thr 10 \
      --data path/to/training_run.pp_Final.nii \
      --discard 100 \
      --out_dir ./existing_output_directory \
      --prefix mask_method

Omit ``--data`` when only the file used as the online ``match_path`` is needed.

Offline outputs
---------------

The command always writes:

``mask_method.masked_templates.nii.gz``
   Input templates after the analysis mask is applied. Use this image to check
   template order and alignment.

``mask_method.template_data.npz``
   Labels, voxel-selection masks, masked template weights, and voxel counts.
   This is the file used as the online ``match_path``.

When ``--data`` is provided, the command additionally writes:

``mask_method.act_traces.npz``
   Label-keyed offline activity traces, including zeros for discarded volumes.

``mask_method.template_pairwise_stats.csv``
   Pairwise selected-voxel overlap and spatial template correlation.

``mask_method.trace_pairwise_stats.csv``
   Pairwise correlations between activity traces after discarded volumes.

``mask_method.traces.png`` and ``mask_method.traces.html``
   Static trace plot and interactive summary with template and trace
   correlation heatmaps.

With ``--save_txt``, the command also writes one ``<label>.act_trace.txt`` file
per template in the current working directory.

Inspect the activity traces to decide on an appropriate
``hit_thr``. Mask scores depend on the template weights, selected voxels, and
data scale, so there is no universal threshold.

2. Configure the online run
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

3. Check the online results
===========================

The real-time run writes the common ESAM score, hit, action, and report files
described in :ref:`output-files`. If the scores do not resemble the offline
traces, check the analysis mask, voxel ordering, preprocessing, and input data
scale before changing the threshold.
