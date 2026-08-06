# rtcog

`rtcog` is a Python package for closed-loop real-time fMRI experiments. It
receives masked voxel data and motion estimates from AFNI, applies a configurable
preprocessing pipeline, and can trigger experimental actions when incoming brain
volumes match selected templates.

The full documentation is available at <https://rtcog.readthedocs.io>.

## Installation

Clone the repository and create either the full environment (including the
PsychoPy participant GUI) or the minimal headless environment.

```bash
git clone https://github.com/nimh-sfim/rtcog.git
cd rtcog

# Full installation
conda env create -f env.yaml
conda activate rtcog
python -m pip install -e .
```

For preprocessing without PsychoPy GUI dependencies:

```bash
conda env create -f minimal_env.yaml
conda activate rtcog_min
python -m pip install -e .
```

Native real-time scanner use also requires AFNI. The full GUI environment uses
PortAudio for audio recording. See the
[installation guide](https://rtcog.readthedocs.io/en/latest/installation.html)
for the Docker option and additional prerequisites.

## Verify the installation

Use the entry point for the environment you installed. It should display its
command-line help without starting an experiment:

```bash
rtcog --help      # Full environment
rtcog_min --help  # Minimal environment
```

An experiment requires a YAML configuration, a mask matching the voxel stream,
the expected number of volumes, and an existing output directory. See the
[usage guide](https://rtcog.readthedocs.io/en/latest/usage.html) for complete
Basic and ESAM examples.
