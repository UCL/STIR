#! /bin/sh
# A script to check to see if reconstruction of listmode data gives the expected result.
#
#  Copyright (C) 2011 - 2011-01-14, Hammersmith Imanet Ltd
#  Copyright (C) 2011-07-01 - 2011, Kris Thielemans
#  Copyright (C) 2014, 2022, 2023 University College London
#  Copyright (C) 2021, University of Pennsylvania
#  This file is part of STIR.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  See STIR/LICENSE.txt for details
#      
# Author Nikos Efthimiou, Kris Thielemans
#

# Scripts should exit with error code when a test fails:
if [ -n "$TRAVIS" -o -n "$GITHUB_WORKSPACE" ]; then
    # The code runs inside Travis or GHA
    set -e
fi

echo This script should work with STIR version ">=" 6.0. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.
echo

#
# Options
#
MPIRUN=""

#
# Parse option arguments (--)
# Note that the -- is required to suppress interpretation of $1 as options 
# to expr
#
while test `expr -- "$1" : "--.*"` -gt 0
do

  if test "$1" = "--mpicmd"
  then
    MPIRUN="$2"
    shift 1
  elif test "$1" = "--help"
  then
    echo "Usage: `basename $0` [--mpicmd somecmd] [install_dir]"
    echo "(where [] means that an argument is optional)"
    echo "See README.txt for more info."
    exit 1
  else
    echo Warning: Unknown option "$1"
    echo rerun with --help for more info.
    exit 1
  fi

  shift 1

done 

if [ $# -eq 1 ]; then
  echo "Prepending $1 to your PATH for the duration of this script."
  PATH=$1:$PATH
fi
echo "Using `command -v OSMAPOSL`"
echo "Using `command -v lm_to_projdata`"

ThereWereErrors=0
ErrorLogs=""

for TOForNOT in "nonTOF" "TOF"; do
    if [ "$TOForNOT" = "nonTOF" ]; then
        # non-TOF data
        export INPUT=PET_ACQ_small.l.hdr.STIR
        export TEMPLATE=Siemens_mMR_seg2.hs
        # Note: test data contains only 612 ms of data, so use a very short frame of 0.5s
        FRAME_DURATION=0.5
        export NORM_PROJDATA=my_acfs_nonTOF_lmtest.hs # only ACFs in this test
    else
        # TOF data
        log=lm_to_projdata_input_formats.log
        lm_to_projdata --input-formats  > ${log} 2>&1
        if ! grep -q ROOT lm_to_projdata_input_formats.log; then
            echo "ROOT support has not been installed in this STIR version."
            echo "Unfortunately, we cannot do the listmode TOF tests at the moment"
            echo "as we don't have a TOF listmode sample file in the distribution."
            continue
        fi
        export INPUT_ROOT_FILE=test_PET_GATE.root
        export INPUT=root_header.hroot
        export TEMPLATE=template_for_ROOT_scanner.hs
        FRAME_DURATION=30
        export NORM_PROJDATA=my_acfs_TOF_lmtest.hs # only ACFs in this test
    fi

    export MAX_SEG_NUM=1
    echo ""
    echo "============= $TOForNOT tests using $INPUT =========="
    echo ""

    # echo "=== Simulate normalisation data"
    # For attenuation data we are going to use a cylinder in the center,
    # with water attenuation values
    echo "=== Generate attenuation image"
    generate_image  lm_generate_atten_cylinder.par
    echo "=== Calculate ACFs"
    log="create_${NORM_PROJDATA}.log"
    calculate_attenuation_coefficients --ACF ${NORM_PROJDATA} my_atten_image.hv ${TEMPLATE} > "$log" 2>&1
    if [ $? -ne 0 ]; then
        echo "ERROR running calculate_attenuation_coefficients. Check ${log}"; exit 1;
    fi

    echo "=== Creating my_test_lm_frame.fdef (time frame definitions)"
    rm -f my_test_lm_frame.fdef
    echo "0 0.1" > my_test_lm_frame.fdef # skip the first .1s, to test if this feature works
    echo "1 $FRAME_DURATION" >> my_test_lm_frame.fdef
    export FRAMES

    for use_frame in true false; do
        if $use_frame; then
            FRAMES=my_test_lm_frame.fdef
            suffix="frame_$TOForNOT"
        else
            FRAMES=""
            suffix="counts_$TOForNOT"
        fi
        echo "=============== Using ${suffix} definition ============="

        # create sinograms
        echo "=== Unlist listmode data (for comparison)"
        logfile=lm_to_projdata_${suffix}.log
        export OUT_PROJDATA_FILE="my_sinogram_${suffix}"
        if lm_to_projdata  lm_to_projdata.par > "$logfile" 2>&1
        then
            echo "---- Executable ran ok"
        else
            echo "---- There were problems here! Check $logfile"
            exit 1 # abort as the rest will fail anyway
            ThereWereErrors=1;
            ErrorLogs="$ErrorLogs $logfile"
        fi

        export ADD_SINO="my_additive_sinogram_${suffix}.hs"
        echo "=== Create additive sino ${ADD_SINO}"
        # Just create a constant sinogram with a value max_prompts/50
        add_value=`list_projdata_info --max "${OUT_PROJDATA_FILE}_f1g1d0b0.hs" 2> /dev/null |grep max|awk -F: '{print $2/50}'`
        logfile=stir_math_add_sino_${suffix}.log
        if stir_math -s --including-first --times-scalar 0 --add-scalar "$add_value" "$ADD_SINO" "${OUT_PROJDATA_FILE}_f1g1d0b0.hs"  > "$logfile" 2>&1
        then
            echo "---- Executable ran ok"
        else
            echo "---- There were problems here! Check $logfile"
            exit 1 # abort as the rest will fail anyway
            ThereWereErrors=1;
            ErrorLogs="$ErrorLogs $logfile"
        fi

        echo "=== Reconstruct listmode data without cache"
        export filename=my_output_t_lm_pr_seg${MAX_SEG_NUM}_${suffix}
        SENS_lm=my_sens_t_lm_pr_seg${MAX_SEG_NUM}.hv
        export cache=0
        export recompute_cache=0
        export recompute_sensitivity=1
        logfile=OSMAPOSL_test_lm_${suffix}_1.log
        if env SENS=${SENS_lm} ${MPIRUN} OSMAPOSL OSMAPOSL_test_lm.par > "$logfile" 2>&1
        then
            echo "---- Executable ran ok"
        else
            echo "---- There were problems here!"
            ThereWereErrors=1;
            ErrorLogs="$ErrorLogs $logfile"
        fi

        echo "=== Reconstruct listmode data with cache and store it on disk"
        # first remove all cached files
        rm -f my_CACHE*bin
        export filename=my_output_t_lm_pr_seg${MAX_SEG_NUM}_${suffix}_with_new_cache
        export cache=5000
        export recompute_cache=1
        export recompute_sensitivity=0
        logfile=OSMAPOSL_test_lm_${suffix}_2.log
        if env SENS=${SENS_lm} ${MPIRUN} OSMAPOSL OSMAPOSL_test_lm.par > "$logfile" 2>&1
        then
            echo "---- Executable ran ok"
        else
            echo "---- There were problems here!"
            ThereWereErrors=1;
            ErrorLogs="$ErrorLogs $logfile"
        fi

        echo "=== Compare reconstructed images with and without caching LM file"
        logfile=my_output_comparison_nocache_vs_new_cache_${suffix}.log
        if compare_image my_output_t_lm_pr_seg${MAX_SEG_NUM}_${suffix}_1.hv my_output_t_lm_pr_seg${MAX_SEG_NUM}_${suffix}_with_new_cache_1.hv > "$logfile" 2>&1
        then
            echo "---- This test seems to be ok !"
        else
            echo "---- There were problems here!"
            ThereWereErrors=1;
            ErrorLogs="$ErrorLogs $logfile"
        fi

        echo "=== Reconstruct listmode data with cache loaded from the disk"
        export filename=my_output_t_lm_pr_seg${MAX_SEG_NUM}_${suffix}_with_old_cache
        export cache=40000
        export recompute_cache=0
        export recompute_sensitivity=0
        logfile=OSMAPOSL_test_lm_${suffix}_3.log
        if env SENS=${SENS_lm} ${MPIRUN} OSMAPOSL OSMAPOSL_test_lm.par > "$logfile" 2>&1
        then
            echo "---- Executable ran ok"
        else
            echo "---- There were problems here!"
            ThereWereErrors=1;
            ErrorLogs="$ErrorLogs $logfile"
        fi

        echo "=== Compare reconstructed images without caching LM file and with loading cache from disk"
        logfile=my_output_comparison_nocache_vs_existing_cache_${suffix}.log
        if compare_image my_output_t_lm_pr_seg${MAX_SEG_NUM}_${suffix}_1.hv my_output_t_lm_pr_seg${MAX_SEG_NUM}_${suffix}_with_old_cache_1.hv > "$logfile" 2>&1
        then
            echo "---- This test seems to be ok !"
        else
            echo "---- There were problems here!"
            ThereWereErrors=1;
            ErrorLogs="$ErrorLogs $logfile"
        fi

        echo "=== Reconstruct projection data for comparison"
        SENS_proj=my_sens_t_proj_seg${MAX_SEG_NUM}_${suffix}.hv
        export filename=my_output_t_proj_seg${MAX_SEG_NUM}_${suffix}
        export recompute_sensitivity=1
        logfile=OSMAPOSL_test_proj_${suffix}.log
        if env SENS="${SENS_proj}" INPUT="${OUT_PROJDATA_FILE}_f1g1d0b0.hs" ${MPIRUN} OSMAPOSL OSMAPOSL_test_proj.par > "$logfile" 2>&1
        then
            echo "---- Executable ran ok"
        else
            echo "---- There were problems here!"
            ThereWereErrors=1;
            ErrorLogs="$ErrorLogs $logfile"
        fi

        echo "=== Compare sensitivity images"
        logfile=my_sens_comparison_${suffix}.log
        if compare_image ${SENS_proj} ${SENS_lm} > "$logfile" 2>&1
        then
            echo "---- This test seems to be ok !"
        else
            echo "---- There were problems here!"
            ThereWereErrors=1;
            ErrorLogs="$ErrorLogs $logfile"
        fi

        echo "=== Compare reconstructed images"
        logfile=my_output_comparison_proj_vs_lm_${suffix}.log
        if compare_image my_output_t_proj_seg${MAX_SEG_NUM}_${suffix}_1.hv my_output_t_lm_pr_seg${MAX_SEG_NUM}_${suffix}_1.hv > "$logfile" 2>&1
        then
            echo "---- This test seems to be ok !"
        else
            echo "---- There were problems here!"
            ThereWereErrors=1;
            ErrorLogs="$ErrorLogs $logfile"
        fi
    done # frame or counts

done # TOForNOT

echo
echo '--------------- End of tests -------------'
echo
if test ${ThereWereErrors} = 1  ;
then
    echo "Check what went wrong. The *.log files might help you, in particular:"
    echo $ErrorLogs
else
    echo "Everything seems to be fine !"
    echo 'You could remove all generated files using "rm -f my_* *.log"'
fi

exit ${ThereWereErrors}
