Transcribing
============

The ESAM participant interface records one audio file per completed hit as
``<prefix>.hit<NNN>.wav``. The optional transcription utility processes those
files with the open-source Whisper package.

Install the optional dependency in a separate virtual environment. It is not
included in either rtcog conda environment:

.. code:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -U openai-whisper

Run the utility from the repository root:

.. code:: bash

   python rtcog/matching/transcribe.py \
     --in_dir /path/to/audio_outputs \
     --out_dir /path/to/transcripts \
     --prefix sub-001_run-01 \
     --model turbo

``--prefix`` must match the run's ``out_prefix``. The utility searches
``in_dir`` for ``<prefix>.hit???.wav``, creates ``out_dir`` when needed, and
writes files such as ``<prefix>.hit001.transcript.txt`` there. It prints the
search pattern and exits without writing files when no audio matches.

``--model`` is optional and defaults to ``turbo``. The first use of a model
downloads its files, so perform that setup before a scanner session on systems
with restricted network access.

Show all options without loading a model:

.. code:: bash

   python rtcog/matching/transcribe.py --help
