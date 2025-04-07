# 03/28/2025 - Marly Rubin
#
# This script starts up AFNI realtime and sets all environment variables.
# If REALTIME_DIR is not provided, it will default to the current working directory.
# -------------------------------
if [ -z "$REALTIME_DIR" ]; then
    echo "REALTIME_DIR is not set. Using current working directory."
    REALTIME_DIR=$(pwd)
fi

echo "Realtime path: $REALTIME_DIR"

cd "$REALTIME_DIR" || { echo "Failed to cd to REALTIME_DIR"; exit 1; }

if ! ls EPIREF+orig* &>/dev/null; then
    echo "Error: EPIREF+orig file does not exist!"
    exit 1
fi

if [ ! -f "GMribbon_R4Feed.nii" ]; then
    echo "Error: GMribbon_R4Feed.nii file does not exist!"
    exit 1
fi

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
