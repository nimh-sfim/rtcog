#!/bin/bash
# 03/28/2025 - Marly Rubin
#
# This script starts up AFNI realtime and sets all environment variables. If -r or --reference is supplied, it will set
# the variables for the reference scan. Otherwise, --nvol/-n can optionally specify the number of volumes (default=1200).
# If REALTIME_DIR is exported, it defaults to the current working directory.
# -------------------------------

USE_REFERENCE=false
NVOL=1200  # Default value

# --- Parse arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -r|--reference) USE_REFERENCE=true ;;
        -n|--nvol)
            shift
            if [[ -z "$1" || ! "$1" =~ ^[0-9]+$ ]]; then
                echo "Error: --nvol/-n requires a numeric argument."
                exit 1
            fi
            NVOL="$1"
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Set directory ---
if [ -z "$REALTIME_DIR" ]; then
    echo "REALTIME_DIR is not set. Using current working directory."
    REALTIME_DIR=$(pwd)
fi

echo "Realtime path: $REALTIME_DIR"
cd "$REALTIME_DIR" || { echo "Failed to cd to REALTIME_DIR"; exit 1; }

echo "Starting AFNI RealTime..."

# --- Set core AFNI environment variables ---
export AFNI_REALTIME_Registration=3D:_realtime
export AFNI_REALTIME_MP_HOST_PORT=localhost:53214
export AFNI_REALTIME_SEND_VER=YES
export AFNI_REALTIME_SHOW_TIMES=YES
export AFNI_REALTIME_Function=FIM
export AFNI_REALTIME_Graph=Realtime

# --- Mode-specific settings ---
if [ "$USE_REFERENCE" = true ]; then
    echo "+++ Setting variables for reference scan"
    export AFNI_REALTIME_Base_Image=2
    export AFNI_REALTIME_Mask_Vals=ROI_means
else
    if [ ! -e "EPIREF+orig.HEAD" ] && [ ! -e "EPIREF+orig.BRIK.gz" ]; then
        echo "Error: EPIREF+orig file does not exist!"
        exit 1
    fi

    if [ ! -f "GMribbon_R4Feed.nii" ]; then
        echo "Error: GMribbon_R4Feed.nii file does not exist!"
        exit 1
    fi

    echo "+++ Using $NVOL volumes (default is 1200 if not specified)"
    echo "+++ Setting variables for rest of experiment"

    export AFNI_REALTIME_Resampling=Quintic
    export AFNI_REALTIME_External_Dataset=EPIREF+orig
    export AFNI_REALTIME_Mask_Dset=GMribbon_R4Feed.nii
    export AFNI_REALTIME_Base_Image=0
    export AFNI_REALTIME_Mask_Vals=All_Data_light
    export AFNI_REALTIME_Duration=$NVOL
fi

# --- Launch AFNI in realtime mode ---
afni -rt &