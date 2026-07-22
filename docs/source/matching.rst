################
Matching methods
################

``rtcog`` supports three built-in spatial matching methods for ESAM mode. Select
the method in the ``matching`` section of your YAML config with ``match_method``.

All three matchers require an input file, which is created offline
(detailed instruction below).

To implement a new matching method, see :doc:`custom_matcher`.

Built-in methods
================

``svr``
   Uses a pretrained support vector regression model. Prepare the model with
   ``rtcog/matching/offline/svr.py`` and pass the resulting pickle file with
   ``--match_path``.

``mask``
   Uses template masks and average masked activity. Prepare the template file
   with ``rtcog/matching/offline/mask.py`` and pass the resulting ``.npz`` file
   with ``--match_path``.

``nmi``
   Uses signed normalized mutual information against template maps.
   Prepare the template file with ``rtcog/matching/offline/nmi.py`` and pass the
   resulting ``.npz`` file with ``--match_path``.

NMI matching
============

The NMI matcher can use any template-map file that can be masked into the same
voxel space as incoming processed TRs.

Input shape determines how templates are read:

- A 3D image is treated as one template.
- A 4D image is treated as multiple templates, with one template per volume in
  file order.

Convert the template maps into an ``rtcog`` template file *before* the real-time
run. If you also provide processed training data with ``--data``, the command
scores that run offline with the same signed-NMI calculation used online:

.. code:: bash

   python rtcog/matching/offline/nmi.py \
      --data path/to/training_data.nii \
      --templates_path path/to/templates.nii \
      --mask path/to/mask.nii \
      --template_labels_path path/to/template_labels.txt \
      --discard 100 \
      --out_dir ./output_directory \
      --prefix prefix

Omit ``--data`` when you only want to prepare the template file and template
statistics.

Template labels are optional:

- If ``--template_labels_path`` is omitted, labels default to ``T01``, ``T02``,
  and so on.
- If labels are provided for a 4D image, they should be comma-separated and in
  the same order as the volumes in the template file.

The output ``prefix.nmi_templates.npz`` contains:

``labels``
   Template labels in template-map order.

``templates``
   Raw masked templates with shape ``(n_templates, n_voxels)``. These are
   required to assign the Pearson-correlation sign.

``template_bins``
   Precomputed binned templates used for the NMI score.

``n_bins``
   Number of bins used for all templates.

The offline command also writes a CSV sidecar with pairwise template statistics:
overlap voxels and spatial Pearson correlation. If ``--data`` is omitted,
spatial correlations are shown as a heatmap in a
``prefix.nmi_template_stats.html`` report. For continuous templates,
selected-mask overlap is based on nonzero voxels, so spatial correlation is
usually the more informative statistic.

When ``--data`` is provided, the offline command also writes:

``prefix.nmi_scores.npy``
   Signed NMI scores with shape ``(n_templates, n_timepoints)``.

``prefix.nmi_raw_scores.npy``
   Unsigned raw ``NMI - 1`` scores before applying the Pearson-correlation sign.

``prefix.nmi_correlations.npy``
   Pearson correlations used to assign the sign of each NMI score.

``prefix.nmi_score_traces.npz``
   Label-keyed signed NMI traces.

``prefix.nmi_scores.png`` and ``prefix.nmi_scores.html``
   Static and interactive score summaries. The HTML report includes spatial
   template-correlation and temporal score-trace correlation heatmaps.

``prefix.nmi_score_pairwise_stats.csv``
   Pairwise Pearson correlations for the signed NMI score traces after
   discarded volumes.

At run time, each processed TR is compared with each template in two steps:

1. The matcher computes Pearson correlation between the raw template and the
   processed TR. The correlation supplies the sign of the NMI score.

   - Mutual information can be high for inverted patterns, so Pearson
     correlation is used to preserve whether a high-NMI pattern is template-like
     or inverted.

2. Templates are scored with binned normalized mutual information. The reported
   score is ``sign(correlation) * max(NMI - 1, 0)``. Zero or non-finite
   correlations receive a score of zero.

   - Positive ``hit_thr`` values only trigger positive template-like matches;
     negative scores remain available for interpreting inverted high-NMI
     patterns.

Configure the run with:

.. code:: yaml

   matching:
     match_method: nmi
     match_start: 100
     vols_noaction: 45

Then pass the template file to ``rtcog``:

.. code:: bash

   rtcog \
      --exp_type esam \
      --match_path path/to/prefix.nmi_templates.npz \
      --hit_thr your_threshold
