# rtfMRI

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/nimh-sfim/rtCAPs.git
cd rtCAPs
```

### 2. Install dependencies

#### Prerequisites

- [portaudio](https://www.portaudio.com/)
- afni (version AFNI_25.0.07)

#### Create environment

```bash
conda env create -f env.yml
```

```bash
conda activate rtcaps
pip install .
```

## Setup

### 1. Update config

This package relies on options specified in a yaml file to run. We've provided you with a default setup at `rtfmri/config/default_config.yaml` which you can customize to your setup. Please note that the order of preprocessing steps is preserved, so any changes to that will affect the pipeline.

### 2. Real-Time Scanner Setup

To run rtCAPs in a live scanner environment, see [guides/scanner_setup.md](guides/scanner_setup.md).

## Customization

### Adding preprocessing steps

You can easily extend the real-time fMRI preprocessing pipeline by defining a new step as a subclass of `PreprocStep`. Each step operates on one TR at a time and integrates into the existing framework.

### 1. **Create your step class**

In `rtfmri/preproc/preproc_steps.py`, define a new class that inherits from `PreprocStep`. Your class must implement the following method:

- `_run(self, pipeline)`: **required**  
  This is where you apply your preprocessing logic. It operates on `pipeline.processed_tr` (a NumPy array of shape `(N_voxels, 1)`) and returns transformed data.

**Optional methods:**

You can optionally implement:

- `_start(self, pipeline)`: initialize the state at the first TR
- `_save(self, pipeline)`: save any extra outputs if `save=True`

Example:

```python
class CustomStep(PreprocStep):
    def _run(self, pipeline):
        new_data = some_function(pipeline.processed_tr) # Apply your transformation
        return new_data
```

**Naming convention**:
Class names ending with "Step" are registered using the lowercase prefix (e.g., `CustomStep` → `"custom"`). If your class does not end with "Step", it is registered using the full class name in lowercase (e.g., `ZScore` → `"zscore"`).

Private classes (classes that start with `_`) are not registered.

### 2. Enable the step in your config file

Add your step to the steps list in your YAML config file, in the order you want it to be applied during preprocessing:

```yaml
steps:
  - name: custom
    enabled: true
    save: false
```

The string "custom" will automatically map to your `CustomStep` class.

### 3. (Optional) Registering with StepTypes

If you want to check whether a step is active in `Pipeline` without relying on string literals, add it to the `StepType` enum:

```python
# step_type.py
class StepType(Enum):
  # ...
  CUSTOM = 'custom'
```

Then in `pipeline.py`:

```python
if StepType.CUSTOM.value in self.steps:
   do_something()
```

---

### Adding matching methods

This software offers two methods for spatial template matching:

`SVRMatcher`: Uses a pretrained SVR model.

`MaskMatcher`: Uses template masks.

You can easily add your own matching method by defining a new step as a subclass of `Matcher`. Each step operates on one TR at a time and integrates into the existing framework.

### 1. **Create your matcher class**

In `rtfmri/matching/matching_methods.py` (or wherever appropriate), define a new class that inherits from Matcher. Your class must implement the following elements in `__init__`:

- Load any models or templates.
- Set the following attributes:
  - `self.input`: The model or template data.
  - `self.template_labels`: List of labels used for scoring.
  - `self.Ntemplates`: Number of templates.
  - `self.func`: Function used to compute matching scores. Must accept (tr_data, input, template_labels) and return scores.

Also be sure to:

- Call `self.setup_shared_memory()` to initialize shared memory buffers.
- Call `self.mp_shm_ready.set()` once your matcher is fully initialized.

Optional override:

- `match()`: You can override this method to customize the matching logic for each TR

Example:

```python
class CustomMatcher(Matcher):
    def __init__(self, match_opts, match_path, Nt, mp_evt_end, mp_new_tr, mp_shm_ready):
        super().__init__(match_opts, match_path, Nt, mp_evt_end, mp_new_tr, mp_shm_ready)
        
        self.input = load_custom_model(match_path)  # Load your templates/model
        self.template_labels = list(self.input["labels"])
        self.Ntemplates = len(self.template_labels)
        self.func = custom_score_function  # Define separately
        
        self.setup_shared_memory()
        self.mp_shm_ready.set()
```

**Naming convention**:
Class names ending with "Matcher" are registered using the lowercase prefix (e.g., `CustomMatcher` → `"custom"`). If your class does not end with "Matcher", it is registered using the full class name in lowercase (e.g., `CustomMethod` → `"custommethod"`).

Private classes (classes that start with `_`) are not registered.

### 2. **Enable the matcher in your config**

Specify the matcher in your YAML config file under the matching section:

```yaml
matching:
  match_method: custom
```

The string "custom" automatically maps to your CustomMatcher class.

---

### Creating your own experiment plugin

`rtcog` includes two built-in experiment types:

- Preproc: Performs basic real-time fMRI preprocessing.
- ESAM (Experience Sampling): Builds on Preproc to support template matching, response collection, and dynamic real-time data streaming.

#### Plugin Components

| Component      | Role                                                                                        |
| -------------- | ------------------------------------------------------------------------------------------- |
| Experiment     | Defines how each fMRI volume is processed                                                   |
| Controller     | Orchestrates the experiment runtime by responding to events and optionally updating the GUI |
| GUI (optional) | Presents stimuli and collects participant responses via Psychopy                            |

If you’re designing a custom experiment, such as an online neurofeedback protocol or novel stimulus design, you can create your own experiment plugin by implementing or extending these components.

#### The Experiment Class

The `Experiment` handles how each TR is processed.

Because preprocessing and template matching are fully configurable via the config file or subclassing `PreprocStep`, `Matcher`, and/or `HitDetector`, subclassing `Experiment` is not recommended. Most use cases can simply reuse one of the following:

- `PreprocExperiment`: Basic real-time fMRI preprocessing.
- `ESAMExperiment`: Extends `PreprocExperiment` to support online template matching, participant response collection, and real-time data visualization.

#### The Controller Class

The `Controller` class is the controller for your experiment. It gives you access to synchronization flags via the `SyncEvents` object (the `self.sync` attribute), allowing you to update the GUI, log events, or trigger feedback based on experiment state:

- `end`: Signals the end of the experiment
- `hit`: Triggered when a TR sufficiently matches a template (ESAM only)
- `qa_end`: Marks the end of a question/response block (ESAM only)

Example for an ESAM experiment:

```python
class MyController(ESAMController):
    def _run(self):
        if self.sync.hit.is_set():
            self.gui.show_custom_prompt()
            self.sync.hit.clear()
            self.sync.qa_end.set()
```

#### The GUI Class (Optional)

The GUI defines what the participant sees and interacts with. You can present:

- Visual prompts
- Trial feedback
- Questions or rating scales
- Audio/voice recording
- Or anything else that Psychopy supports

You can inherit from:

| Class        | Description                                                     |
| ------------ | --------------------------------------------------------------- |
| `BaseGUI`    | Blank starting point                                            |
| `PreprocGUI` | Displays a fixation cross and general instructions              |
| `EsamGUI`    | Adds voice recording, question prompts, and response collection |

```python
class MyGUI(EsamGUI):
    def show_custom_prompt(self):
        stim = TextStim(win=self.ewin, text='Hello', pos=(0.0,0.06), bold=True),
        self._draw_stims(stim)
```

#### Registering Your Custom Experiment

To make your experiment available to `rtcog`, register it in `rtfmri/experiment_registry.py`:

```python
"my_custom_experiment": {
    "experiment": ESAMExperiment, # Or PreprocExperiment
    "controller" MyController,
    "gui": MyGUI,                 # Optional
}
```

Now, you can pass the name of your experiment when running `rtcog` and it will look it up in the registry: `-exp_type my_custom_experiment`

---

## Transcribing

To transcribe the hit audio files, you first need to create a basic virtual environment due to some issues using the one that comes with this software.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U openai-whisper
```

The first time you run the script it will have to download the model, which takes a few minutes.

To run the script:

```bash
python rtfmri/matching/transcribe.py -i <input_dir> -o <output_dir> -p <input_prefix> -m <model>
```