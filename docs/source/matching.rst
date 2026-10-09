################
Matching methods
################

``rtcog`` supports four built-in spatial matching methods for ESAM mode. Each
method compares a processed volume with one or more templates and returns one
score per template. Select the method with ``matching.match_method`` and pass
its offline-prepared model or template file as ``match_path``.

Method guides
=============

.. toctree::
   :maxdepth: 1

   matching/svr
   matching/mask
   matching/pearson
   matching/nmi

``svr``
   Uses one pretrained linear support vector regression model per template.
   Training requires a previously processed run.

``mask``
   Computes weighted average activity within thresholded template masks.
   A processed run is optional and can be used to preview score traces.

``pearson``
   Computes spatial Pearson correlation between each template and the current
   processed volume. A processed run is optional for offline evaluation.

``nmi``
   Computes signed normalized mutual information (NMI) between each template
   and the current processed volume. A processed run is optional for offline
   evaluation.

Shared inputs
=============

All methods require template maps and an analysis mask during offline
preparation. The real-time run must use the same mask space and voxel ordering.
Template maps may be a single 3D image or a 4D image with one template per
volume.

.. _template-label-file:

Template label file
-------------------

Template labels are stored in a text file as one comma-separated line, in the
same order as the template volumes, with no header. For example:

.. code:: text

   dmn,visual,somatosensory

The label file is required by the mask and SVR procedures. Pearson and NMI use
``T01``, ``T02``, and so on when it is omitted.

Shared runtime settings
=======================

A minimal method selection looks like this:

.. code:: yaml

   matching:
     match_method: pearson
     match_start: 100
     vols_noaction: 45

``match_start`` is the first volume index at which matching can begin.
``vols_noaction`` is the cooldown after an action ends before another hit can
be declared. The top-level ``discard`` setting controls initial online
preprocessing, while the offline commands' ``--discard`` option applies only to
their training or evaluation data.

Pass the method-specific file identified in its guide as ``match_path`` and set
``hit_thr`` from representative offline scores. See :doc:`usage` for the full
run command and :ref:`output-files` for the files written by an ESAM run.

To implement a new matching method, see :doc:`custom_matcher`.
