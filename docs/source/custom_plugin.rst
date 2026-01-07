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

The ``Processor`` handles how each TR is processed.

Because preprocessing and template matching are fully configurable via
the config file or subclassing ``PreprocStep``, ``Matcher``, and/or
``HitDetector``, subclassing ``Processor`` is **not** recommended. Most use
cases can simply reuse one of the following:

- ``BasicProcessor``: Basic real-time fMRI preprocessing.
- ``ESAMProcessor``: Extends ``BasicProcessor`` to support online
  template matching and real-time data visualization.

The ActionSeries Class (Optional)
---------------------------------

The ``ActionSeries`` class responds to the state of the experiment. By
extending ``BaseActionSeries``, you can implement your own custom logic
for what should occur at each stage of the experiment:

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

   class MyActionSeries(BaseActionSeries):
       def __init__(self):
           gui = MyGUI(opts=opts)
           super().__init__(sync, opts=opts, gui=gui)
           
       def on_start(self):
           startup_function()
           self.gui.draw_resting_screen()
       def on_loop(self):
           poll_for_escape_key()
       def on_hit(self):
           self.gui.show_custom_prompt()
           apply_stimulation()
       def on_end(self):
           teardown_function()
           self.gui.close_psychopy_infrastructure()

The GUI Class (Optional)
------------------------

The ``GUI`` defines what the participant sees and interacts with.

If you only want to change the Likert questions displayed to the participant, you can
simply create a json file with your custom questions and put the path in your config yaml
file under ``q_path``. Define the text, labels, and name for each question: 

.. code-block:: json

    {
        "text": "Q1/11. How alert were you?",
        "labels": ["Fully asleep", "Somewhat sleepy", "Somewhat alert", "Fully alert"],
        "name": "alert"
    }

See ``questions_v1.json`` for a full example.

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
- Or anything else that ``Psychopy`` supports

Make sure to instantiate your ``GUI`` as an attribute of your
``ActionSeries``.

Registering Your Custom Experiment Plugin
-----------------------------------------

To make your plugin available to ``rtcog``, register it in
``rtcog/experiment_registry.py``:

.. code:: python

   "my_custom_experiment": {
       "processor": ESAMProcessor, # Or BasicProcessor
       "action" MyActionSeries     # Optional
   }

Now, you can pass the name of your experiment plugin when running ``rtcog`` and
it will look it up in the registry: ``--exp_type my_custom_experiment``
