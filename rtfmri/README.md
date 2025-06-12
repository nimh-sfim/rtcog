# rtFMRI

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
```

## Customoization

### Adding preprocessing steps

You can easily extend the real-time fMRI preprocessing pipeline by defining a new step as a subclass of `PreprocStep`. Each step operates on one TR at a time and integrates into the existing framework.

### 1. **Create your step class**

In `preproc_steps.py`, define a new class that inherits from `PreprocStep`. Your class must implement the following method:

- `_run(self, pipeline)`: **required**  
  This is where you apply your preprocessing logic. It operates on `pipeline.processed_tr` (a NumPy array of shape `(N_voxels, 1)`) and returns transformed data.

**Optional methods:**

You can optionally implement:

- `_start(self, pipeline)`: initalize the state at the first TR
- `_save(self, pipeline)`: save any extra outputs if `save=True`

Example:

```python
class CustomStep(PreprocStep):
    def _run(self, pipeline):
        new_data = some_function(pipeline.processed_tr) # Apply your transformation
        return new_data
```

Your class name must end with Step, and it will automatically be registered using the lowercase name (e.g., "custom").

### 2. Enable the step in your config file

Add your step to the steps list in your YAML config file, in the order you want it to be applied during preprocessing:

```yaml
steps:
  - name: custom
    enabled: true
    save: false
```

The string "custom" will be matched to your CustomStep class via automatic registration.

### 3. (Optional) Add your step to `StepTypes` enum

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
