Start AFNI in real-time mode
============================

AFNI's realtime plugin can be configured completely through environment
variables, so the RT Options window does not need to be configured by hand.
Set the reference dataset, mask, and expected run length, then start AFNI from
the **same terminal**:

.. code:: bash

    # Values that change between experiments.
    RT_EPI_REF="/absolute/path/to/EPIREF+orig"
    RT_MASK="/absolute/path/to/GMribbon_R4Feed.nii"
    RT_NVOLS=1200

    # Realtime registration and display.
    export AFNI_REALTIME_Registration=3D:_realtime
    export AFNI_REALTIME_Resampling=Cubic
    export AFNI_REALTIME_Reg_Base_Mode=External_Dataset
    export AFNI_REALTIME_Base_Image=0
    export AFNI_REALTIME_External_Dataset="${RT_EPI_REF}[0]"
    export AFNI_REALTIME_Graph=Realtime
    export AFNI_REALTIME_NR="${RT_NVOLS}"

    # Data sent to rtcog.
    export AFNI_REALTIME_Mask_Dset="${RT_MASK}"
    export AFNI_REALTIME_Mask_Vals=All_Data_light
    export AFNI_REALTIME_MP_HOST_PORT=localhost:53214
    export AFNI_REALTIME_SEND_VER=YES
    export AFNI_REALTIME_SHOW_TIMES=YES
    export AFNI_REALTIME_Function=FIM

    afni -rt

Use absolute paths so startup does not depend on the current directory. For an
AFNI ``HEAD/BRIK`` dataset, provide its prefix (for example,
``/path/to/EPIREF+orig``), not the ``.HEAD`` filename. The ``[0]`` selector
chooses volume 0 of the external reference dataset. The base-image variable is
also set to ``0`` so that the plugin default is explicit. For an external
dataset, the ``[0]`` selector is what determines the actual reference volume;
AFNI ignores ``AFNI_REALTIME_Base_Image`` for that selection.

AFNI reads the environment when it starts. If the reference or mask is created
after AFNI has already been launched, quit AFNI and run the block above again
after both datasets exist. Changing exports in another terminal does not update
an AFNI process that is already running.

No manual RT Options setup is required. The external-reference and mask
variables act as runtime overrides and might not appear as selected datasets in
the GUI. In particular, AFNI intentionally does not display a dataset supplied
through ``AFNI_REALTIME_External_Dataset`` in the ``Extern Dset`` field. A blank
dataset chooser therefore does not mean that the override is unset. Before
starting AFNI, the datasets can be checked with:

.. code:: bash

    3dinfo -exists "${AFNI_REALTIME_External_Dataset}"
    3dinfo -exists "${AFNI_REALTIME_Mask_Dset}"

Each command prints ``1`` when AFNI can resolve the dataset.

These variable names and accepted values are documented in AFNI's
`README.environment <https://afni.nimh.nih.gov/pub/dist/doc/program_help/README.environment.html>`_.
