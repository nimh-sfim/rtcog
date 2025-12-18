# rtcog

This project contains the code to perform realtime fMRI experiments in the following form:

1. The system monitors brain activity on a TR-by-TR basis
2. If the subject brain looks like one of a subset of templates, then the program fires a survey
3. The program continues monitoring the brain in the background.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/nimh-sfim/rtcog.git
cd rtcog
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
conda activate rtcog
pip install . # or `pip install -e .` for editable mode
```


