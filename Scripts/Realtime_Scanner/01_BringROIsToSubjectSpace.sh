set -e
if [ $# -lt 3 ]; then
 echo -e "\x1B[0;34mUsage: $basename $0 EPIREFFILE ANATREF ROIFILE [MNITEMPLATE]\x1B[0m"
 echo -e "\x1B[0;34mUsage: $basename $0 asd015_epi_run1+orig asd015_anat+orig roi11+tlrc []\x1B[0m"
 exit
fi
EPIREF=$1
ANATREF=$2
ROIFILE=$3
if [ $# -eq 4 ]; then
	MNITEMPLATE=$4
else
	AFNIHOME=`which afni | awk -F / '{$NF=""}1' | tr -s ' ' '/'`
	MNITEMPLATE=`echo ${AFNIHOME}MNI152_2009_template.nii.gz`
        MNITEMPLATE_SSW=`echo ${AFNIHOME}MNI152_2009_template_SSW.nii.gz`
        MNITEMPLATE_GM=/data/SFIMJGC/PRJ_rtCAPs/Others/MNI152_2009_template_GM.nii.gz
        DEFAULT_TEMPLATE=1

        3dcalc -overwrite -a ${MNITEMPLATE_SSW}'[4]' -expr 'a' -prefix ${MNITEMPLATE_GM}
fi
EPI_NAME=`echo "${EPIREF}" | awk -F '+' '{print $1}'`
SBJ=`echo "${EPI_NAME}" | awk -F '_' '{print $1}'`
echo "${EPIREF} --> ${EPI_NAME}  --> ${SBJ}"
#cd /data/SFIMJGC/PRJ_rtCAPs/PrcsData/${SBJ}/D01_Realtime/
echo "++ Using file [${MNITEMPLATE}] as the common space reference..."

### # (0) Deoblique the reference volume
### # ==================================
### echo "++ Deoblique the reference volume..."
### echo "===================================="
### 3dWarp -deoblique -overwrite -prefix ${EPI_NAME}.deoblique.nii ${EPIREF}

# (1) Extract first volume for the EPI
# ------------------------------------
echo ""
echo "++ Extracting first volume for EPI reference file..."
echo "===================================================="
3dcalc -overwrite -a ${EPIREF}'[3]' -expr 'a' -prefix EPIREF

# (2) Automask the EPI ref volume
# -------------------------------
echo ""
echo "++ Automask for EPI reference volume..."
echo "======================================="
3dAutomask -overwrite -prefix EPIREF.mask.nii EPIREF+orig

# (3) Skull-strip the ref volume
# ------------------------------
echo ""
echo "++ Create skull stripped version of the EPI reference volume..."
echo "==============================================================="
3dcalc -overwrite -a EPIREF+orig -m EPIREF.mask.nii -expr 'a*m' -prefix EPIREF.ns
3dAutobox -overwrite -prefix EPIREF.mask.autobox.nii EPIREF.mask.nii
3dZeropad -overwrite -I -1 -S -2 -prefix EPIREF.mask.autobox.nii EPIREF.mask.autobox.nii
3dresample -overwrite -master EPIREF.mask.nii -inset EPIREF.mask.autobox.nii -prefix EPIREF.mask_R4Feed.nii

# (4) Create Bias Map
# -------------------
echo ""
echo "++ Create EPI bias map...."
echo "=========================="
#3dcalc -overwrite -a EPIREF+orig. -expr '1' -prefix ALLVOL
3dBlurInMask -overwrite -FWHM 20 -mask EPIREF.mask.nii -input EPIREF+orig. -prefix EPIREF.bias

EPIBIAS_min=`3dROIstats -minmax -quiet -nomeanout -mask EPIREF.mask.nii EPIREF.bias+orig | awk '{print $1}'`
EPIBIAS_max=`3dROIstats -minmax -quiet -nomeanout -mask EPIREF.mask.nii EPIREF.bias+orig | awk '{print $2}'`
3dcalc -datum float -overwrite -a EPIREF.bias+orig. -expr "(a-${EPIBIAS_min})/(${EPIBIAS_max}-${EPIBIAS_min})" -prefix EPIREF.bias.scaled.nii
#3dcalc -overwrite -a EPIREF+orig. -b EPIREF.bias.scaled.nii -m EPIREF.mask.nii -expr 'm*a/b' -prefix EPIREF.ns+orig

#rm ALLVOL+orig.*

# (4) Create intracranial mask for anatomical
# -------------------------------------------
echo ""
echo "++ Skull Striping..."
echo "===================="
3dSkullStrip -overwrite -prefix ANAT.mask -input ${ANATREF}
3dcalc       -overwrite -a ANAT.mask+orig. -expr 'step(a)' -overwrite -prefix ANAT.mask
3dcalc       -overwrite -a ANAT.mask+orig -b ${ANATREF} -expr 'a*b' -prefix ANAT.bc.ns

# (5) Bias correct the anatomical
# -------------------------------
echo ""
echo "++ Run 3dSeg for additional bias correction of MP-RAGE..."
echo "========================================================="
if [ -d Segsy ]; then rm -rf Segsy; fi
3dSeg -anat ANAT.bc.ns+orig -mask ANAT.mask+orig
3dcopy -overwrite Segsy/AnatUB+orig. ANAT.bc+orig
3dcalc -overwrite -a ${ANATREF} -b ANAT.bc+orig -m ANAT.mask+orig -expr 'm*(b*step(b))' -prefix ANAT.bc.ns

# (6) Compute transformation from MP-RAGE space into MNI space and vice-versa
# -----------------------------------------------------------------------
echo ""
echo "++ Convert to MNI space..."
echo "=========================="
@auto_tlrc -overwrite -base ${MNITEMPLATE}  -input ANAT.bc.ns+orig -no_ss -twopass
cat_matvec -ONELINE ANAT.bc.ns+tlrc::WARP_DATA > MNI2Anat.Xaff12.1D
cat_matvec -ONELINE MNI2Anat.Xaff12.1D -I      > Anat2MNI.Xaff12.1D
rm ANAT.bc.ns_WarpDrive.log
rm ANAT.bc.ns.Xaff12.1D
rm ANAT.bc.ns.Xat.1D

# (7) Compute transformation between EPI reference volume and MP-RAGE
# -------------------------------------------------------------------
echo ""
echo "++ Computing alignment between EPI and MPRAGE..."
echo "================================================"
3dZeropad -I 20 -S 20 -prefix ANAT.bc.ns -overwrite ANAT.bc.ns+orig
align_epi_anat.py -anat ANAT.bc.ns+orig                       \
                    -epi  EPIREF.ns+orig                      \
                    -epi_base 0 -anat2epi -anat_has_skull no  \
                    -deoblique on                             \
                    -epi_strip None         \
                    -suffix _al2EPI                            \
                    -giant_move                               \
                    -master_anat SOURCE -overwrite

3dAutobox -overwrite -prefix ANAT.bc.ns -input ANAT.bc.ns+orig
3dAutobox -overwrite -prefix ANAT.bc    -input ANAT.bc+orig
3dAutobox -overwrite -prefix ANAT.bc.ns_al2EPI -input ANAT.bc.ns_al2EPI+orig

3dcalc     -overwrite -a ANAT.bc.ns_al2EPI+orig. -expr 'step(a)' -prefix ${SBJ}_Anat.mask.FBfromANAT.nii
3dresample -overwrite -rmode NN -master ${EPIREF} -prefix ${SBJ}_Anat.mask.FBfromANAT.nii -inset ${SBJ}_Anat.mask.FBfromANAT.nii

mv ANAT.bc.ns_al2EPI_mat.aff12.1D Anat2REF.Xaff12.1D
if [ -f ANAT.bc.ns_al2EPI_e2a_only_mat.aff12.1D ]; then rm ANAT.bc.ns_al2EPI_e2a_only_mat.aff12.1D; fi
#if [ -f ANAT.bc.ns_al+orig.HEAD ];             then rm ANAT.bc.ns_al+orig.*; fi

# (8) Compute all additional transformations
# ------------------------------------------
echo ""
echo "++ Generate all necessary transformation matrices..."
echo "===================================================="
cat_matvec -ONELINE Anat2REF.Xaff12.1D -I > REF2Anat.Xaff12.1D
cat_matvec -ONELINE Anat2REF.Xaff12.1D \
                    MNI2Anat.Xaff12.1D >    MNI2REF.Xaff12.1D
cat_matvec -ONELINE MNI2REF.Xaff12.1D -I >  REF2MNI.Xaff12.1D

# (9) Bring ROIs to EPI reference space
# -------------------------------------
echo ""
echo "++ Bring ROI to EPI space..."
echo "============================"
ROIFILEPREFIX=`echo ${ROIFILE} | awk -F'+' '{print $1}' | awk -F'.' '{print $1}'` 
3dAllineate -final linear -overwrite \
            -input          ${ROIFILE}         \
            -1Dmatrix_apply MNI2REF.Xaff12.1D  \
            -master         EPIREF+orig        \
            -prefix         ${ROIFILEPREFIX}_R4Feed.nii

if [ ${DEFAULT_TEMPLATE} -gt 0 ]; then
  3dAllineate -final NN -overwrite \
              -input ${MNITEMPLATE_GM} \
              -1Dmatrix_apply MNI2REF.Xaff12.1D \
              -master EPIREF.mask.nii \
              -prefix GMribbon_R4Feed.nii
fi

3dcalc -overwrite \
       -a GMribbon_R4Feed.nii \
       -b EPIREF.mask_R4Feed.nii \
       -c ${ROIFILEPREFIX}_R4Feed.nii'[0]' \
       -expr 'a*b*step(abs(c))' \
       -prefix GMribbon_R4Feed.nii

# Alternative GM
3dcalc -a ./Segsy/Classes+orig. -expr 'equals(a,2)' -prefix rm.GMribbon_C_R4Feed.nii
3dmask_tool -input rm.GMribbon_C_R4Feed.nii -prefix rm.GMribbon_C_R4Feed.nii -overwrite -dilate_inputs 2 -2
3dAllineate -final NN -input rm.GMribbon_C_R4Feed.nii -1Dmatrix_apply Anat2REF.Xaff12.1D -master EPIREF+orig. -prefix GMribbon_C_R4Feed.nii
3dcalc -a GMribbon_C_R4Feed.nii -b EPIREF.mask_R4Feed.nii -c 'Frontier2013_CAPs_R4Feed.nii[0]' -expr 'a*b*step(abs(c))' -prefix GMribbon_C_R4Feed.nii -overwrite
rm rm.GMribbon_C_R4Feed.nii
#3dAllineate -final NN -overwrite \
#            -input          ${ROIFILEPREFIX}_ZSCORED.nii         \
#            -1Dmatrix_apply MNI2REF.Xaff12.1D  \
#            -master         EPIREF+orig        \
#            -prefix         ${ROIFILEPREFIX}_ZSCORED_R4Feed.nii

