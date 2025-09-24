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

.. code-block:: bash

   conda env create -f env.yml

   conda install rtcog
   pip install .  # or `pip install -e .` for editable mode
