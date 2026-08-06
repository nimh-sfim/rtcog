Installation
============

Requirements
------------

The supported installation paths are:

- **Full environment**: preprocessing plus the PsychoPy participant GUI and
  audio support.
- **Minimal environment**: preprocessing and matching without the PsychoPy
  participant GUI.
- **Minimal Docker image**: the same headless entry point in the AFNI-based
  container defined by this repository's ``Dockerfile``.

Native scanner use requires `AFNI <https://afni.nimh.nih.gov/>`_. The Docker
image pins the exact AFNI image used by the container. The full environment also
requires `PortAudio <https://www.portaudio.com/>`_ for audio recording. Conda is
required for the environment files below; Docker is only required for the
container workflow.

1. Clone the repository
-----------------------

.. code-block:: bash

   git clone git@github.com:nimh-sfim/rtcog.git
   cd rtcog

2. Create an environment
------------------------

For access to all of rtcog's features, install as normal:

.. code-block:: bash

   conda env create -f env.yaml

   conda activate rtcog
   python -m pip install -e .

If you do not require rtcog's GUI features, you can install a
minimal version instead:

.. code-block:: bash

   conda env create -f minimal_env.yaml

   conda activate rtcog_min
   python -m pip install -e .

This version does not have Psychopy GUI presentation and will only
run preprocessing and matching. The ``rtcog_min`` command does not start the
participant GUI or the live operator streaming process. This is useful for
headless deployments and testing.

3. Verify the installation
--------------------------

The installed entry points should display their help without starting an
experiment:

.. code-block:: bash

   rtcog --help
   rtcog_min --help

Use the entry point for the environment you installed. Continue to :doc:`usage`
for the required inputs and complete run commands.

Minimal Docker image
--------------------

Alternatively, you can use the Docker image for the minimal version.
We use this image for testing the real-time preprocessing pipeline
locally and on HPC systems.

Build the image from the repository root:

.. code-block:: bash

   docker build --platform linux/amd64 -t rtcog .


After building the Docker image, run a smoke test:

.. code:: bash

   docker run --rm --platform linux/amd64 rtcog

The image's default command is ``rtcog_min --help``, so this smoke test exits
after displaying the CLI help. See :ref:`docker-usage` for an actual run.
