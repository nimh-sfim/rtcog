Installation
============

1. Clone the repository
-----------------------

.. code-block:: bash

   git clone https://github.com/nimh-sfim/rtCAPs.git
   cd rtCAPs

2. Install dependencies
-----------------------

* `portaudio <https://www.portaudio.com/>`_
* afni (version AFNI_25.0.07)
* conda

3. Create environment
---------------------

.. code-block:: bash

   conda env create -f env.yml

   conda install rtcaps
   pip install .  # or `pip install -e .` for editable mode
