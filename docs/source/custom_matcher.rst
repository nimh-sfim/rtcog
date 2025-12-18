#######################
Adding matching methods
#######################

This software offers two methods for spatial template matching:

* ``SVRMatcher``: Uses a pretrained SVR model.
* ``MaskMatcher``: Uses template masks.

You can easily add your own matching method by defining a new step as a
subclass of ``Matcher``. Each step operates on one TR at a time and
integrates into the existing framework.

1. **Create your matcher class**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``rtcog/matching/matching_methods.py`` (or wherever appropriate),
define a new class that inherits from Matcher. Your class must implement
the following:


- ``_match(self, tr_data)``: **required**
  This method performs the actual matching computation for the current
  TR and returns a 1D NumPy array of length Ntemplates containing
  the match scores for each template at the current TR.

During initalization, your matcher must:

- Set ``self.template_labels``: List of template labels used for scoring.
- Set ``self.Ntemplates``: Number of templates.
- Call ``self.setup_shared_memory()`` to initialize shared memory
  buffers.
- Call ``self.mp_shm_ready.set()`` once your matcher is fully
  initialized. This allows for integration with the streaming process.

If needed, you can load any templates or models it needs from a filepath.
Add ``--match_path <filepath>`` when running `rtcog`.


Example:

.. code:: python

   class CustomMatcher(Matcher):
       def __init__(self, match_opts, Nt, sync, match_path):
        super().__init__(match_opts, Nt, sync, match_path)
           
           self.input = load_custom_model(match_path)  # Load your templates/model
           self.template_labels = list(self.input["labels"])
           self.Ntemplates = len(self.template_labels)
           
           self.setup_shared_memory()
           self.mp_shm_ready.set()

        def _match(self, tr_data):
            # Implement your matching logic here
            scores = compute_custom_scores(self.input, tr_data)
            return scores  

**Naming convention**: Class names ending with “Matcher” are registered
using the lowercase prefix (e.g., ``CustomMatcher`` → ``"custom"``). If
your class does not end with “Matcher”, it is registered using the full
class name in lowercase (e.g., ``CustomMethod`` → ``"custommethod"``).

Private classes (classes that start with ``_``) are not registered.

1. **Enable the matcher in your config**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specify the matcher in your YAML config file under the matching section:

.. code:: yaml

   matching:
     match_method: custom

The string “custom” automatically maps to your ``CustomMatcher`` class because
of the naming convention.