############
NMI matching
############

NMI matching measures nonlinear spatial similarity using binned normalized
mutual information. Because mutual information alone does not distinguish a
template from its inverse, Pearson correlation supplies the sign of each
score.

See :doc:`/matching` for the inputs and settings shared by all matching
methods.

Workflow summary
================

1. Prepare the masked and binned templates, optionally scoring a representative
   processed run in the same command.
2. Review the generated files and reports.
3. Use the prepared ``.npz`` file as ``match_path`` in an ESAM run.

1. Prepare and evaluate offline
================================

The output directory must already exist. ``--data`` and
``--template_labels_path`` are optional. When ``--data`` is supplied,
``--discard`` leaves that many initial volumes unscored.

.. code:: bash

   python rtcog/matching/offline/nmi.py \
      --templates_path path/to/templates.nii \
      --mask path/to/mask.nii \
      --template_labels_path path/to/template_labels.txt \
      --data path/to/training_run.pp_Final.nii \
      --discard 100 \
      --out_dir ./existing_output_directory \
      --prefix nmi_method

Omit ``--data`` when only the runtime template file and template comparison
report are needed. The optional processed run should use the same mask and
preprocessing planned for the ESAM run.

2. Review the offline outputs
=============================

The command always writes:

``nmi_method.nmi_templates.npz``
   Labels, masked templates, precomputed template bins, and the bin count. Use
   this file as the online ``match_path``.

``nmi_method.nmi_template_pairwise_stats.csv``
   Pairwise template overlap counts and spatial Pearson correlations.

With no ``--data``, it also writes
``nmi_method.nmi_template_stats.html``, an interactive template-correlation
summary.

When ``--data`` is supplied, the report is instead expanded with these files:

``nmi_method.nmi_scores.npy``
   Signed NMI scores with shape ``(n_templates, n_timepoints)``.

``nmi_method.nmi_raw_scores.npy``
   Unsigned ``NMI - 1`` values before the correlation sign is applied.

``nmi_method.nmi_correlations.npy``
   Pearson correlations used to determine score signs.

``nmi_method.nmi_score_traces.npz``
   Signed scores stored as one named array per template label.

``nmi_method.nmi_score_pairwise_stats.csv``
   Pairwise correlations between signed score traces after discarded volumes.

``nmi_method.nmi_scores.png`` and ``nmi_method.nmi_scores.html``
   Static trace plot and interactive report with template and trace correlation
   heatmaps.

The reported score is ``sign(correlation) * max(NMI - 1, 0)``. A positive
``hit_thr`` therefore selects template-like patterns, while inverted patterns
remain visible as negative scores. Zero or non-finite correlations produce a
score of zero.

3. Configure the online run
===========================

Add the method to the complete ESAM configuration:

.. code:: yaml

   match_path: path/to/nmi_method.nmi_templates.npz
   hit_thr: your_threshold

   matching:
     match_method: nmi
     match_start: 100
     vols_noaction: 45

The online analysis mask must preserve the voxel count and ordering used during
offline preparation. The real-time outputs are described in
:ref:`output-files`.
