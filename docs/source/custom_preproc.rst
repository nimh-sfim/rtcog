Creating preprocessing steps
=============================

You can easily extend the real-time fMRI preprocessing pipeline by
defining a new step as a subclass of ``PreprocStep``. Each step operates
on one TR at a time and integrates into the existing framework.

1. Create your step class
------------------------------

In ``rtcog/preproc/preproc_steps.py``, define a new class that inherits
from ``PreprocStep``. Your class must implement the following method:

- ``_run(self, pipeline)``: **required**
  This is where you apply your preprocessing logic. It operates on
  ``pipeline.processed_tr`` (a NumPy array of shape ``(N_voxels, 1)``)
  and returns transformed data.

**Optional methods:**

You can optionally implement:

- ``_start(self, pipeline)``: initialize the state at the first TR
- ``_save(self, pipeline)``: save any extra outputs if ``save=True``

Example:

.. code:: python

   class CustomStep(PreprocStep):
       def _run(self, pipeline):
           new_data = some_function(pipeline.processed_tr) # Apply your transformation
           return new_data

**Naming convention**: Class names ending with “Step” are registered
using the lowercase prefix (e.g., ``CustomStep`` → ``"custom"``). If
your class does not end with “Step”, it is registered using the full
class name in lowercase (e.g., ``ZScore`` → ``"zscore"``).

Private classes (classes that start with ``_``) are not registered.

2. Enable the step in your config file
---------------------------------------

Add your step to the steps list in your YAML config file, in the order
you want it to be applied during preprocessing:

.. code:: yaml

   steps:
     - name: custom
       enabled: true
       save: false

The string “custom” will automatically map to your ``CustomStep`` class.

3. (Optional) Registering with StepTypes
-----------------------------------------

If you want to check whether a step is active in ``Pipeline`` without
relying on string literals, add it to the ``StepType`` enum:

.. code:: python

   # step_type.py
   class StepType(Enum):
     # ...
     CUSTOM = 'custom'

Then in ``pipeline.py``:

.. code:: python

   if StepType.CUSTOM.value in self.steps:
      do_something()
