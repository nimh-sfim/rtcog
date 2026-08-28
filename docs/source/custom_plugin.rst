Creating an experiment plugin
=============================


``rtcog`` includes two built-in experiment types:

- **Basic**: Performs basic real-time fMRI preprocessing.
- **ESAM** (Experience Sampling): Builds on Basic to support template
  matching, response collection, and dynamic real-time data streaming.

Plugin Components
-----------------

If you’re designing a custom experiment, such as an online neurofeedback
protocol or novel stimulus design, you can create your own experiment
plugin by implementing or extending these components:

+-------------------------+--------------------------------------------+
| Component               | Role                                       |
+=========================+============================================+
| Processor               | Defines how each fMRI volume is processed  |
+-------------------------+--------------------------------------------+
| ActionSeries (Optional) | Performs actions based on experiment state |
+-------------------------+--------------------------------------------+

The Processor Class
-------------------

The ``Processor`` coordinates how each TR moves through the processing workflow.
Most experiments should reuse one of the existing processor classes:

- ``BasicProcessor``: Basic real-time fMRI preprocessing.
- ``ESAMProcessor``: Extends ``BasicProcessor`` to support online
  template matching and real-time data visualization.

Customize the individual parts of that workflow through their own configuration
and extension points:

- **Preprocessing (``PreprocStep``):** Select, order, and configure steps in the
  ``steps`` section of the YAML file. For a new preprocessing operation,
  subclass ``PreprocStep`` as described in
  :doc:`Creating preprocessing steps <custom_preproc>`, then enable it in YAML.
- **Template matching (``Matcher``, ESAM only):** Select and configure a matching
  method in the ``matching`` section of the YAML file. For a new matching
  algorithm, subclass ``Matcher`` as described in
  :doc:`Adding matching methods <custom_matcher>`.
- **Hit detection (``HitDetector``, ESAM only):** Configure thresholds, consecutive
  volumes, and motion rejection with ``hit_thr`` and the ``hits`` section of the
  YAML file. Subclass ``HitDetector`` when the detection algorithm itself must
  change.

Do not add any of these experiment-specific behaviors by modifying or subclassing
``BasicProcessor`` or ``ESAMProcessor``. The processor should only select the
overall processing mode, while customization belongs in the dedicated class or YAML
configuration above.

The ActionSeries Class (Optional)
---------------------------------

The ``ActionSeries`` class responds to the state of the experiment. For example,
it can display a survey when a hit is detected in ESAM mode. By extending
``BaseActionSeries``, you can implement your own custom logic for what should
occur at each stage of the experiment:

- ``on_start()``: The beginning of the experiment
- ``on_loop()``: Main experiment loop
- ``on_hit()``: Triggered when a TR sufficiently matches a template
  (ESAM only)
- ``on_end()``: The end of the experiment

``ActionSeries`` are optional. If you don’t provide one, the experiment
will simply run without performing any additional actions. You can also
pass ``--no_action`` when running ``rtcog`` to prevent your
``ActionSeries`` from running.

``rtcog`` by default comes with two action series:

- ``BasicActionSeries``: Displays a basic GUI until the experiment
  ends
- ``ESAMActionSeries``: Also collects voice recording and question
  responses at each hit

If you have a ``GUI`` (outlined below), it should be owned by your ``ActionSeries`` so it can
be updated throughout the experiment.

Example for an ESAM experiment:

.. code:: python

   from rtcog.controller.action_series import BaseActionSeries

   class MyActionSeries(BaseActionSeries):
       def __init__(self, sync, opts):
           gui = MyGUI(opts=opts)
           super().__init__(sync, opts=opts, gui=gui)
           
       def on_start(self):
           startup_function()
           self.gui.draw_resting_screen()
       def on_loop(self):
           poll_for_escape_key()
       def on_hit(self):
           self.gui.show_custom_prompt()
       def on_end(self):
           teardown_function()
           self.gui.close_psychopy_window()

The GUI Class (Optional)
------------------------

The ``GUI`` defines what the participant sees and interacts with.

If you only want to change the Likert questions displayed to the participant,
create a JSON file with your custom questions and put its path in the YAML
configuration under ``q_path``.

The file must contain a JSON list. Every question requires ``text`` and ``name``
fields, and question names should be unique because responses are keyed by name.
``labels`` is optional; omitting it uses the default five-point agreement scale.

.. code-block:: json

    [
        {
            "text": "Q1/1. How alert were you?",
            "labels": ["Fully asleep", "Somewhat sleepy", "Somewhat alert", "Fully alert"],
            "name": "alert"
        }
    ]

See the :download:`questions_v1.json example
<../../rtcog/resources/questions_v1.json>` for a complete questionnaire.

However, if you want to create a more complex GUI, you can
create a custom ``GUI`` class.

You can inherit from:

+----------------+-----------------------------------------------------------+
| Class          | Description                                               |
+================+===========================================================+
| ``BaseGUI``    | Blank starting point                                      |
+----------------+-----------------------------------------------------------+
| ``BasicGUI``   | Displays a fixation cross and general instructions        |
+----------------+-----------------------------------------------------------+
| ``EsamGUI``    | Adds voice recording, question prompts, and response      |
|                | collection                                                |
+----------------+-----------------------------------------------------------+

Example:

.. code:: python

   class MyGUI(EsamGUI):
       def show_custom_prompt(self):
           self._draw_stims(self._custom_stim)

You can
present:

- Visual prompts
- Trial feedback
- Questions or rating scales
- Audio/voice recording
- Or anything else that ``PsychoPy`` supports

Make sure to instantiate your ``GUI`` as an attribute of your
``ActionSeries``.

Registering Your Custom Experiment Plugin
-----------------------------------------

To make your plugin available to ``rtcog``, import the custom classes and add an
entry in ``rtcog/experiment_registry.py`` after ``EXPERIMENT_REGISTRY`` is
defined:

.. code:: python

   from my_experiment import MyActionSeries

   EXPERIMENT_REGISTRY["my_custom_experiment"] = {
       "processor": ESAMProcessor, # Or BasicProcessor
       "action": MyActionSeries,   # Optional
   }

Now, you can pass the name of your experiment plugin when running ``rtcog`` and
it will look it up in the registry: ``--exp_type my_custom_experiment``
