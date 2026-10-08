=====================
Contributing to rtcog
=====================

Thank you for helping improve ``rtcog``!

Before you start
================

- Search the `issue tracker <https://github.com/nimh-sfim/rtcog/issues>`_ for
  related work.
- For a large feature or architectural change, open an issue first so the
  approach and scope can be discussed.

Set up a development environment
================================

Clone the repository and create one of the supported Conda environments.

For preprocessing, matching, documentation, and most tests on Apple or Linux:

.. code-block:: bash

   git clone https://github.com/nimh-sfim/rtcog.git
   cd rtcog
   conda env create -f minimal_env.yaml
   conda activate rtcog_min
   python -m pip install -e .

For PsychoPy participant-interface or audio work on Apple-silicon macOS:

.. code-block:: bash

   conda env create -f env.yaml
   conda activate rtcog
   python -m pip install -e .

Create a focused branch for your work:

.. code-block:: bash

   git switch -c descriptive-branch-name

Repository map
==============

``rtcog/comm/``
   AFNI real-time communication.

``rtcog/preproc/``
   Configurable preprocessing pipeline and steps.

``rtcog/matching/``
   Online matching, hit detection, and offline preparation.

``rtcog/processor/``
   Basic and ESAM volume-processing workflows.

``rtcog/controller/``
   Experiment lifecycle and triggered actions.

``rtcog/gui/``
   PsychoPy participant interfaces.

``rtcog/viz/``
   Operator displays and reports.

``rtcog/config/``
   Config files.

``rtcog/resources/``
   Questions, images, and sounds used at runtime.

``tests/``
   Automated tests.

``docs/source/``
   Sphinx/Read the Docs source.

The installed commands are ``rtcog`` for the full application and
``rtcog_min`` for operation without GUI dependencies.

Make a change
=============

- Follow the style of the surrounding code. Public APIs should have clear
  NumPy-style docstrings.
- Add or update tests for behavior changes and bug fixes.
- Update user documentation and examples when configuration, commands,
  outputs, or public behavior changes.
- Do not commit generated output, ``docs/_build/``, logs, local environments,
  or research data.

Configuration changes
---------------------

When changing a config option:

1. Update parsing and validation in ``rtcog/utils/options.py``.
2. Update the relevant example configuration in ``rtcog/config/``.
3. Update ``docs/source/options.rst`` and any affected usage examples.
4. Add coverage in ``tests/test_options.py`` or the relevant subsystem test.

Extending ``rtcog``
----------------

See the customization guides in ``docs/source/customize.rst``.

Run the tests
=============

The complete testing guide is in ``docs/source/testing.rst``. It explains how
to add focused tests, run the continuous-integration selection, and perform
snapshot testing.

Before opening a pull request, run the standard selection:

.. code-block:: bash

   python -m pytest -q -m "not snapshot"

Update the documentation
========================

User and API documentation is written in reStructuredText under
``docs/source/``. Edit the relevant page, and add new pages to a ``toctree``—
usually in ``docs/source/index.rst`` or a section landing page so they appear in
the published navigation.

Install the documentation dependencies and build with warnings treated as
errors:

.. code-block:: bash

   python -m pip install -r docs/rtd-requirements.txt
   python -m sphinx -W --keep-going -b html docs/source docs/_build/html

Open ``docs/_build/html/index.html`` locally to review the result. Public Python
APIs are included through Sphinx autodoc, so confirm that changed signatures
and docstrings render correctly.

Documentation is hosted at
`rtcog.readthedocs.io <https://rtcog.readthedocs.io>`_. GitHub Actions validates
both the Sphinx build and the standard test suite on pushes and pull requests.

Open a pull request
===================

Before submitting:

.. code-block:: bash

   git status
   git diff --check
   python -m pytest -q -m "not snapshot"
   python -m sphinx -W --keep-going -b html docs/source docs/_build/html

In the pull request:

- Explain the problem and the chosen solution.
- Describe user-visible or configuration changes.
- Include screenshots for meaningful GUI or visualization changes.

Keep review follow-ups in the same branch. A pull request should be small enough
to review confidently and should leave the test and documentation builds green.
