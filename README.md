# rtcog

`rtcog` is a Python package for closed-loop real-time fMRI experiments of the following form:

1. The computer receives volumes from the AFNI real-time plugin on a TR-by-TR basis
2. Each volume is put through a configurable preprocessing pipeline as it arrives
3. If the subject's brain looks like a brain configuration of interest, then the program can
fire a stimulus to the subject (e.g. a survey)
4. The program continues monitoring the brain in the background.

The full documentation is available at <https://rtcog.readthedocs.io>.

## Installation

rtcog supports three installation options:

- **Full environment**: includes the PsychoPy participant GUI.
- **Minimal environment**: runs preprocessing and matching without the GUI.
- **Minimal Docker image**: packages the minimal environment as a Docker image.

Note: The full environment is for macOS ARM only. On Linux, you may use the
minimal environment or the Docker container.

### Quick install (full environment on macOS ARM)

```bash
git clone https://github.com/nimh-sfim/rtcog.git
cd rtcog
conda env create -f env.yaml
conda activate rtcog
python -m pip install -e .
```

``rtcog`` also requires AFNI. The full GUI environment uses
PortAudio for audio recording. See the [installation
guide](https://rtcog.readthedocs.io/en/latest/installation.html) for detailed
instructions.

## Verify the installation

```bash
rtcog --help      # Full environment
rtcog_min --help  # Minimal environment
```

An experiment requires a YAML configuration, a mask matching the voxel stream,
the expected number of volumes, and an existing output directory. See the
[usage guide](https://rtcog.readthedocs.io/en/latest/usage.html) for complete
examples.
