Installation
============

1. Clone the repository
-----------------------

.. code-block:: bash

   git clone git@github.com:nimh-sfim/rtcog.git
   cd rtcog

2. Install dependencies
-----------------------

* `portaudio <https://www.portaudio.com/>`_
* afni (version AFNI_25.0.07)
* conda
* Docker, if using the Minimal mode Docker image

3. Create environment
---------------------

For access to all of rtcog's features, install as normal:

.. code-block:: bash

   conda env create -f env.yaml

   conda activate rtcog
   pip install .  # or `pip install -e .` for editable mode

If you do not require rtcog's GUI features, you can install a
minimal version instead:

.. code-block:: bash

   conda env create -f minimal_env.yaml

   conda activate rtcog_min
   pip install .  # or `pip install -e .` for editable mode

This version does not have Psychopy GUI presentation and will only
run the real-time preprocessing pipeline. This is useful for testing
purposes.

Alternatively, you can use the Docker image for the minimal version.
We use this image for testing the real-time preprocessing pipeline
locally and on HPC systems.

Build the image from the repository root:

.. code-block:: bash

   docker build --platform linux/amd64 -t rtcog .


After building the Docker image, run a smoke test:

.. code:: bash

   docker run --rm --platform linux/amd64 rtcog
