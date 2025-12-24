########
Overview
########

``rtcog`` is an open-source Python package for real-time fMRI (rtfMRI) experiments.
It is designed to monitor spontaneous neural activity as data are acquired and to
trigger adaptive experimental actions based on detected brain states. The software is
intended for use in closed-loop neuroimaging paradigms, including thought monitoring,
neurofeedback, and other online cognitive or clinical applications.

During an active scan, ``rtcog``:

- Receives fMRI volumes from the scanner in real time
- Applies a configurable preprocessing pipeline (e.g., smoothing, normalization)

Then optionally:

- Computes similarity between each incoming TR and predefined brain-state templates
- Registers a *hit* when a TR sufficiently matches a target template and triggers
  experiment-specific actions (e.g., surveys)

Purpose and Scope
-----------------

With traditional experimental paradigms, it is difficult to study the link between
spontaneous brain activity and self-driven covert cognition due to the lack of
objective measures to capture behavioral correlates. ``rtcog`` offers a means to
explore this by allowing researchers to monitor spontaneous activation of brain
configurations of interest as they unfold and use these configurations to trigger
introspective surveys to interrogate the contents of ongoing cognition.

While this motivation guided the original design of ``rtcog``, the software is
intentionally general-purpose and can be applied to a wide range of rtfMRI use cases.

Performance and Architecture
----------------------------

To satisfy real-time constraints, ``rtcog`` employs a multiprocessing architecture
with 3 different processes:

- The preprocessing engine, which operates on incoming TRs
- The participant-facing GUI, which presents surveys or feedback
- The operator interface, which presents data as it arrives for real-time monitoring

Shared memory and synchronization events are used to minimize overhead and ensure
consistent state across processes.


On an M4 Macbook Pro, ``rtcog`` achieves preprocessing latencies of ~0.15 seconds
per fMRI volume after data receipt from AFNI, supporting compatibility with rapid
acquisition protocols.

.. figure:: /_static/images/software_architecture.png
   :alt: rtcog software architecture
   :align: center
   :width: 90%

   Overview of the rtcog multiprocessing architecture.

The preprocessing engine
------------------------

``rtcog`` builds on AFNI’s real-time fMRI infrastructure, which streams image data
directly from the scanner on an TR-by-TR basis. AFNI performs initial processing,
including DICOM-to-NIfTI conversion and online volume registration. Each acquisition
is then transmitted to ``rtcog`` via a socket connection for further processing.

The Pipeline
^^^^^^^^^^^^

Within ``rtcog``, incoming volumes pass through a configurable preprocessing
pipeline that may include spatial smoothing, normalization, and other user-defined
operations. The preprocessing pipeline is modular and extensible, allowing
researchers to tailor processing steps to specific experimental requirements without
modifying the core runtime logic.

Pattern Matching and Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After preprocessing, each volume is compared against user-defined activation
templates to detect neural patterns of interest. ``rtcog`` currently provides two
pattern-matching approaches:

- **Region-of-interest–based matching**, which evaluates activation *levels*
  within specified regions.
- **SVR-based matching**, which uses a support vector machine trained on
  previous data to detect activation *patterns*.

A *hit* is registered when the similarity between a TR and a target template exceeds
user-defined criteria (e.g., the score exceeds a certain threshold for a specified
duration). Hits can be used to trigger downstream experimental logic, such as survey
presentation or feedback delivery.

.. figure:: /_static/images/minimal_software_arch.png
   :alt: preprocessing engine
   :align: center
   :width: 90%

   Detailed view of the preprocessing engine.


Experimental Control and Visualization
--------------------------------------

When a hit is detected (i.e., when a TR matches a target template), ``rtcog`` can
automatically initiate experimental actions during ongoing scanning. Survey
presentation and participant response logging are handled through integration with
**PsychoPy**, enabling time-locked introspective or behavioral probes.

For real-time monitoring and quality assurance, ``rtcog`` integrates with the
**HoloViz** visualization framework to provide live displays of neural signals,
pattern-matching scores, and experiment state.

.. figure:: /_static/images/operator_interface.png
   :alt: operator interface
   :align: center
   :width: 90%

   Screenshot of the operator's view, which is updated in real time as new data comes
   in. The interface consists of three parts: (a) The match (similarity) score
   between each brain volume and each template; (b) The subject's brain activity 
   each hit; (c) The subject's behavioral survey responses at each hit.

Extensibility and Use Cases
---------------------------

``rtcog`` is designed to be extensible at multiple levels, including preprocessing,
pattern matching, and experiment control. New processing steps, matching algorithms,
and experiment modes can be added with minimal changes to existing code. 

This allows ``rtcog`` to support a wide range of real-time neuroimaging applications,
including:

- Monitoring spontaneous thought and internal cognition
- Real-time neurofeedback
- Online neuromodulation and clinical research applications

For more information on how to extend ``rtcog``, refer to :doc:`customize`.
