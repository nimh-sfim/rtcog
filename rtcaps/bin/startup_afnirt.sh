# 03/28/2025 - Marly Rubin
#
# This script starts up AFNI realtime and sets all environment variables.
# Requires REALTIME_DIR to be exported.
# -------------------------------
if [ -z "$REALTIME_DIR" ]; then
    echo "Error: REALTIME_DIR is not set. Exiting..."
    exit 1
fi

echo "Realtime path: $REALTIME_DIR"
cd $REALTIME_DIR || { echo "Failed to cd to REALTIME_DIR"; exit 1; }

echo "Starting AFNI RealTime..."

export AFNI_REALTIME_Registration=3D:_realtime
export AFNI_REALTIME_Resampling=Quintic
export AFNI_REALTIME_Base_Image=0
export AFNI_REALTIME_Graph=Realtime
export AFNI_REALTIME_MP_HOST_PORT=localhost:53214
export AFNI_REALTIME_SEND_VER=YES
export AFNI_REALTIME_SHOW_TIMES=YES
export AFNI_REALTIME_Mask_Vals=All_Data_light
export AFNI_REALTIME_Function=FIM
export AFNI_REALTIME_External_Dataset=EPIREF+orig
export AFNI_REALTIME_Mask_Dset=GMribbon_R4Feed.nii
export AFNI_REALTIME_NR=101

afni -rt &
