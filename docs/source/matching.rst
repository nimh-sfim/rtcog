################
Matching methods
################

``rtcog`` supports two built-in spatial matching methods for ESAM mode. Each
method follows the same overall workflow:

1. Run ``rtcog`` in Basic mode with a training rest run.
2. Run the method's offline command, passing the ``rtcog``-processed training run, to prepare its input file and generate evaluation results.
3. Review the offline results and choose an appropriate hit threshold.
4. Supply the prepared file as ``match_path`` and select the method with
   ``match_method`` in the run configuration.

The method-specific pages below separate the offline preparation and online run
instructions. To implement a new method, see :doc:`custom_matcher`.

Choose a method
===============

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Method
     - Score
     - Prepared online input
   * - :doc:`SVR <matching/svr>`
     - Prediction from one trained linear SVR per template.
     - Pickled model dictionary created by ``offline/svr.py``.
   * - :doc:`Mask <matching/mask>`
     - Average activity in weighted template masks.
     - Template-data ``.npz`` created by ``offline/mask.py``.

Method workflows
================

.. toctree::
   :maxdepth: 1

   matching/svr
   matching/mask

.. _template-label-file:

Create the template label file
==============================

The offline commands use ``--template_labels_path`` to associate a readable
name with each template. Both SVR and mask preparation require this file.

The file contains one comma-separated line with no header. For example, a
``template_labels.txt`` file for three template volumes could contain:

.. code:: text

   dmn,visual,somatosensory

The first label names the first template volume, the second label names the
second volume, and so on. The number and order of labels should match the
template volumes in ``templates_path``.

Configure volume timing
=======================

The volume numbers shown in the method guides are examples, not fixed values.
Configure the timing of the real-time run in your YAML file:

.. code:: yaml

   discard: 10

   matching:
     match_method: svr
     match_start: 100
     vols_noaction: 45

``discard``
   Number of volumes at the beginning of the run that are received and stored
   but excluded from preprocessing. This is a YAML setting and can
   also be overridden on the CLI with ``--discard``.

``match_start``
   Zero-based volume number at which real-time matching begins. Set it to a
   value greater than or equal to ``discard`` so matching does not begin during
   the discarded volumes. Configure it in the YAML ``matching`` section.


``vols_noaction``
   Number of volumes to wait after an action ends before another hit can start
   a new action (a cooldown period). Configure it in the YAML ``matching`` section.

Choose these values for the timing and design of your experiment. The example
values ``10``, ``100``, and ``45`` may all be changed.

Offline ``--discard`` is separate
---------------------------------

The offline preparation commands for SVR and mask matching also accept a
``--discard`` option. It controls how many initial volumes of the training data
are excluded from offline fitting or scoring; it does not set ``discard`` for
the later real-time run. The ``--discard 100`` values in the method guides are
examples and may be changed for your training data.

Choose a hit threshold
======================

SVR and mask scores have method- and dataset-specific distributions. There is
no threshold that works for every method or dataset.

Use representative processed data from the same acquisition and preprocessing
setup to inspect offline score traces. Then choose these settings together:

``hit_thr``
   Score a template must meet or exceed to be counted as a hit.

``nconsec_vols``
   Number of consecutive volumes that must meet the threshold. Configure it in
   the YAML ``hits`` section.

``nonline``
   Maximum number of templates that may meet the threshold simultaneously.
   Configure it in the YAML ``hits`` section.

Keep the voxel space consistent
===============================

The online mask and SVR inputs must be prepared with the same analysis mask and
voxel ordering used by the real-time run.

See :doc:`usage` for a complete ESAM run command and :ref:`output-files` for
the files written during a real-time run.
