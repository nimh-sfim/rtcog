Transcribing
============

To transcribe the hit audio files, you first need to create a basic
virtual environment due to some issues using the one that comes with
this software.

.. code:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -U openai-whisper

The first time you run the script it will have to download the model,
which takes a few minutes.

To run the script:

.. code:: bash

   python rtcog/matching/transcribe.py -i <input_dir> -o <output_dir> -p <input_prefix> -m <model>
