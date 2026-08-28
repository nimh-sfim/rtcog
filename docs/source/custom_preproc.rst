Creating preprocessing steps
=============================

You can easily extend the real-time fMRI preprocessing pipeline by
defining a new step as a subclass of ``PreprocStep``. Each step operates
on one TR at a time and integrates into the existing framework.

1. Create your step class
------------------------------

In ``rtcog/preproc/custom_steps.py``, define a new class that inherits
from ``PreprocStep``. Your class
must implement the following method:

- ``_run(self, pipeline)``: **required**
  This is where you apply your preprocessing logic. It operates on
  ``pipeline.processed_tr`` (a NumPy array of shape ``(N_voxels, 1)``)
  and returns transformed data.

**Optional methods:**

You can optionally implement:

- ``_start(self, pipeline)``: initialize runtime-dependent state when the first
  TR is received
- ``_save(self, pipeline)``: save any extra outputs if ``save=True``

``_start`` is called once, after the first volume arrives and the pipeline has
initialized its run-specific state. Use it for setup that depends on information
available only through the live pipeline, such as the current run state or the
complete set of configured preprocessing steps.

At this point, the first volume has not yet been copied into
``pipeline.Data_FromAFNI`` or assigned to ``pipeline.processed_tr``. If setup
depends on the first volume's voxel values, perform it during the first call to
``_run`` instead.

Example:

.. code:: python

   # rtcog/preproc/custom_steps.py
   from rtcog.preproc.preproc_steps import PreprocStep

   class CustomStep(PreprocStep):
       def _start(self, pipeline):
           # Optional: initialize state that depends on the live pipeline.

       def _run(self, pipeline):
           new_data = some_function(pipeline.processed_tr)
           return new_data

       def _save(self, pipeline):
           # Optional: save any extra results after processing is complete.
           pass

``_run`` must return a NumPy array with the same ``(N_voxels, 1)`` shape as
``pipeline.processed_tr``. 

**Naming convention**: Class names ending with “Step” are registered
using the lowercase prefix (e.g., ``CustomStep`` → ``"custom"``). If
your class does not end with “Step”, it is registered using the full
class name in lowercase (e.g., ``ZScore`` → ``"zscore"``).

Private classes (classes that start with ``_``) are not registered.

2. Enable the step in your config file
---------------------------------------

Add your step to the steps list in your YAML config file using the registered name,
in the order you want it to be applied during preprocessing:

.. code:: yaml

   steps:
     - name: custom
       enabled: true
       save: false

The string “custom” will automatically map to your ``CustomStep`` class.
