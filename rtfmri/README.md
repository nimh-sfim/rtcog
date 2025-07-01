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

The string "custom" will automatically map to your CustomStep class.

### 3. (Optional) Registering with StepTypes

If you want to check whether a step is active in `Pipeline` without relying on string literals, add it to the `StepType` enum:

```python
# step_type.py
class StepType(Enum):
  # ...
  CUSTOM = 'custom'
```

Then in pipeline.py:

```python
if StepType.CUSTOM.value in self.steps:
   do_something()
```
