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
the following elements in ``__init__``:

- Load any models or templates.
- Set the following attributes:

  - ``self.input``: The model or template data.
  - ``self.template_labels``: List of labels used for scoring.
  - ``self.Ntemplates``: Number of templates.
  - ``self.func``: Function used to compute matching scores. Must accept
    (tr_data, input, template_labels) and return scores.

Also be sure to:

- Call ``self.setup_shared_memory()`` to initialize shared memory
  buffers.
- Call ``self.mp_shm_ready.set()`` once your matcher is fully
  initialized.

Optional override:

- ``match()``: You can override this method to customize the matching
  logic for each TR

Example:

.. code:: python

   class CustomMatcher(Matcher):
       def __init__(self, match_opts, match_path, Nt, mp_evt_end, mp_new_tr, mp_shm_ready):
           super().__init__(match_opts, match_path, Nt, mp_evt_end, mp_new_tr, mp_shm_ready)
           
           self.input = load_custom_model(match_path)  # Load your templates/model
           self.template_labels = list(self.input["labels"])
           self.Ntemplates = len(self.template_labels)
           self.func = custom_score_function  # Define separately
           
           self.setup_shared_memory()
           self.mp_shm_ready.set()

**Naming convention**: Class names ending with “Matcher” are registered
using the lowercase prefix (e.g., ``CustomMatcher`` → ``"custom"``). If
your class does not end with “Matcher”, it is registered using the full
class name in lowercase (e.g., ``CustomMethod`` → ``"custommethod"``).

Private classes (classes that start with ``_``) are not registered.

2. **Enable the matcher in your config**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specify the matcher in your YAML config file under the matching section:

.. code:: yaml

   matching:
     match_method: custom

The string “custom” automatically maps to your CustomMatcher class.