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

You can easily extend the real-time fMRI preprocessing pipeline by defining a new step as a subclass of `PreprocStep`. Each step operates on one TR at a time and integrates seamlessly into the existing framework.

### 1. **Create your step class**

In `preproc_steps.py`, define a new class that inherits from `PreprocStep`. Your class must implement the following method:

- `run(self, pipeline)`: **required**  
  This is where you apply your preprocessing logic. It must read from `pipeline.processed_tr` (a NumPy array of shape `(N_voxels, 1)`) and overwrite it with the transformed data.

**Optional methods:**

If your step generates data that should be saved, you can optionally implement:

- `initialize_array(self, pipeline)`: define any arrays needed before the first TR
- `run_discard_volumes(self, pipeline)`: handle discard TRs
- `save_nifti(self, pipeline)`: save outputs to `.nii` file if `save=True`

Example:

```python
class CustomStep(PreprocStep):
    def run(self, pipeline):
        data = pipeline.processed_tr
        new_data = some_function(data) # Apply your transformation
        pipeline.processed_tr = new_data
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
