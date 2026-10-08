=============
Testing rtcog
=============

The test suite has three categories:

- Ordinary unit and integration tests run in continuous integration.
- Tests marked ``local_data`` require datasets outside a normal checkout.
- The AFNI snapshot test uses committed synthetic inputs and requires a local
  AFNI realtime session.

Add new tests
=============

Add focused tests for new behavior and bug fixes under ``tests/``. Usually,
add the test to the existing file for that subsystem, such as
``tests/test_pipeline.py``. If no suitable file exists, create one named
``test_<subject>.py``. Name each test function ``test_<behavior>`` so pytest
discovers it automatically; tests do not need to be registered elsewhere.

Keep fixtures used by only one test module in that module. Put fixtures shared
by several modules in ``tests/conftest.py``. Prefer pytest's temporary-path and
mocking fixtures, such as ``tmp_path`` and ``monkeypatch``, over writing to the
repository or calling external services.

Run a new test by its pytest node ID while developing:

.. code-block:: bash

   python -m pytest -q tests/test_pipeline.py::test_name

Then run its complete test file and the standard suite before opening a pull
request.

Run the standard suite
======================

Run the same selection used by continuous integration:

.. code-block:: bash

   python -m pytest -q -m "not local_data and not snapshot"

During development, run the narrowest relevant test first, for example:

.. code-block:: bash

   python -m pytest -q tests/test_pipeline.py
   python -m pytest -q tests/test_matcher.py

Tests marked ``local_data`` require datasets that are not part of a normal
checkout. Enable them with ``--local-data`` only after preparing the required
inputs described by the test or its subsystem documentation.

Run the snapshot test
=====================

The snapshot test runs rtcog end to end with a synthetic fMRI volume series. It
compares the output of each saved preprocessing step with a committed reference
to detect unintended numerical changes across the complete realtime pipeline.
It's useful to run when big, overarching changes are made to the orchestration
of preprocessing.

The synthetic volumes contain intensity changes and small deterministic
rigid-body transforms. The first volume is stationary; later volumes include
sub-voxel translations and sub-degree rotations so the snapshot exercises the
six iGLM motion regressors as well as polynomial detrending.

The repository includes the simulation inputs and expected pipeline output:

``tests/data/synthetic_snapshot_input.nii.gz``
   A 24-volume mathematical phantom.

``tests/data/synthetic_snapshot_mask.nii.gz``
   The phantom's 552-voxel mask.

``tests/data/synthetic_snapshot_expected.npz``
   The expected output from every saved preprocessing step.

``rtcog/config/test_default_config.yaml``
   The complete Basic-mode run configuration for these fixtures.


Prerequisites
-------------

Install AFNI and the ``rtcog_min`` development environment described in
:doc:`installation`. The workflow uses the three workspaces under
``Simulation/``:

- ``Simulation/Realtime`` runs AFNI's realtime plugin.
- ``Simulation/Laptop`` runs rtcog.
- ``Simulation/Scanner`` replays the synthetic volumes with ``rtfeedme``.

Open three terminals in the repository root, then follow the directory-specific
instructions below. Each terminal defines ``SNAPSHOT_ROOT`` independently
because shell variables are not shared between terminals.

Terminal 1: start AFNI
----------------------

Enter the realtime workspace, configure AFNI's realtime plugin, and start AFNI.

.. code-block:: bash

   cd Simulation/Realtime
   SNAPSHOT_ROOT="$(cd ../.. && pwd)"
   export AFNI_REALTIME_Registration='3D:_realtime'
   export AFNI_REALTIME_Resampling=Cubic
   export AFNI_REALTIME_Reg_Base_Mode=External_Dataset
   export AFNI_REALTIME_Base_Image=0
   export AFNI_REALTIME_External_Dataset="${SNAPSHOT_ROOT}/tests/data/synthetic_snapshot_input.nii.gz[0]"
   export AFNI_REALTIME_Graph=Realtime
   export AFNI_REALTIME_NR=24
   export AFNI_REALTIME_Mask_Dset="${SNAPSHOT_ROOT}/tests/data/synthetic_snapshot_mask.nii.gz"
   export AFNI_REALTIME_Mask_Vals=All_Data_light
   export AFNI_REALTIME_MP_HOST_PORT=localhost:53214
   export AFNI_REALTIME_SEND_VER=YES
   export AFNI_REALTIME_SHOW_TIMES=YES
   export AFNI_REALTIME_Function=FIM
   afni -rt

Terminal 2: start rtcog
-----------------------

Enter the laptop workspace, activate the development environment, and start
rtcog. The command-line path overrides keep the committed configuration usable
from this workspace. Wait until rtcog prints ``Ready to go``.

.. code-block:: bash

   cd Simulation/Laptop
   SNAPSHOT_ROOT="$(cd ../.. && pwd)"
   conda activate rtcog_min
   rtcog_min \
      -c "$SNAPSHOT_ROOT/rtcog/config/test_default_config.yaml" \
      --mask_path "$SNAPSHOT_ROOT/tests/data/synthetic_snapshot_mask.nii.gz" \
      --out_dir "$SNAPSHOT_ROOT/Simulation/outputs" \
      --snapshot_dir "$SNAPSHOT_ROOT/Simulation/outputs"

Terminal 3: feed the data
-------------------------

Enter the scanner workspace and send the synthetic data to AFNI:

.. code-block:: bash

   cd Simulation/Scanner
   SNAPSHOT_ROOT="$(cd ../.. && pwd)"
   rtfeedme "$SNAPSHOT_ROOT/tests/data/synthetic_snapshot_input.nii.gz"

Compare the result
------------------

After ``rtcog_min`` finishes and writes
``Simulation/outputs/new_snapshots.npz``, return to the repository root and
compare that output with the committed reference:

.. code-block:: bash

   cd "$SNAPSHOT_ROOT"
   python -m pytest -q tests/test_snapshots.py --snapshot

The pytest option includes tests marked ``snapshot``. Snapshot tests remain
outside the normal CI selection because they require a local AFNI session.

The committed reference was recorded with AFNI 25.0.07. If a run with another
AFNI release differs, include ``afni -ver`` with the failure report and review
the AFNI registration output before deciding whether the baseline should move.

Extend or update a snapshot
===========================

The committed snapshot covers all preprocessing steps enabled by
``rtcog/config/test_default_config.yaml``. To test another configuration, run
the same three-terminal workflow with that configuration. With ``snapshot``
enabled, rtcog writes ``new_snapshots.npz`` to the configured
``snapshot_dir``. Add a focused pytest test if the new snapshot should become a
permanent regression check.

Do not update a committed reference merely to make a failing test pass. First
confirm that the difference is an intended result of a code, configuration, or
dependency change. Record the AFNI and dependency versions used to generate a
new reference so future differences can be investigated.

To cover a new or custom step, enable ``save`` for that step and add its key to
``SNAPSHOT_KEYS`` in ``tests/test_snapshots.py``. A custom step can expose
additional internal state by overriding ``snapshot()``:

.. code-block:: python

   def snapshot(self):
       values = super().snapshot()
       values["custom_filter_state"] = self.filter_state
       return values
