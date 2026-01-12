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
