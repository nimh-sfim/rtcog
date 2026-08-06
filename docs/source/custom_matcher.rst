#######################
Adding matching methods
#######################

For the built-in spatial matching methods, see :doc:`matching`. If you want a
different way of deciding when a template matches the current TR, define a
subclass of ``Matcher``.

1. **Create your matcher class**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a new class that inherits from Matcher. Your class must implement
the following:


- ``_match(self, tr_data)``: **required**
  This method performs the actual matching computation for the current
  TR. It must return a 1D NumPy array of length ``Ntemplates`` containing
  the match scores for each template at the current TR.

During initialization, your matcher must:

- Set ``self.template_labels``: List of template labels used for scoring.
- Set ``self.Ntemplates``: Number of templates.
- Call ``self.setup_shared_memory()`` to initialize shared memory
  buffers.
- Call ``self.mp_shm_ready.set()`` once your matcher is fully
  initialized. This allows for integration with the streaming process.

If needed, load templates or models from ``match_path``. Put the subclass in
``rtcog/matching/matcher.py`` or ensure its module is imported before
``Matcher.from_name`` is called; importing the class performs its automatic
registration.


Example:

.. code:: python

   import numpy as np

   from rtcog.matching.matcher import Matcher

   class CustomMatcher(Matcher):
       def __init__(self, match_opts, Nt, sync, match_path):
           super().__init__(match_opts, Nt, sync, match_path)
           
           self.input = load_custom_model(match_path)  # Load your templates/model
           self.template_labels = list(self.input["labels"])
           self.Ntemplates = len(self.template_labels)
           
           self.setup_shared_memory()
           self.mp_shm_ready.set()

       def _match(self, tr_data):
           scores = compute_custom_scores(self.input, tr_data)
           return np.asarray(scores)

**Naming convention**: Class names ending with “Matcher” are registered
using the lowercase prefix (e.g., ``CustomMatcher`` → ``"custom"``). If
your class does not end with “Matcher”, it is registered using the full
class name in lowercase (e.g., ``CustomMethod`` → ``"custommethod"``).

Private classes (classes that start with ``_``) are not registered.

2. **Enable the matcher in your config**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specify the matcher in your YAML config file under the matching section
using the registered name:

.. code:: yaml

   matching:
     match_method: custom

The string “custom” automatically maps to your ``CustomMatcher`` class because
of the naming convention.
