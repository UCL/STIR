#! /bin/sh
# A script to check to see if reconstruction of listmode data gives the expected result.
#
#  Copyright (C) 2011 - 2011-01-14, Hammersmith Imanet Ltd
#  Copyright (C) 2011-07-01 - 2011, Kris Thielemans
#  Copyright (C) 2014, 2022 University College London
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

echo This script should work with STIR version ">=" 5.1. If you have
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

# echo "=== Simulate normalisation data"
# For attenuation data we are going to use a cylinder in the center,
# with water attenuation values
echo "=== Generate attenuation image"
generate_image  lm_generate_atten_cylinder.par
echo "=== Calculate ACFs"
calculate_attenuation_coefficients --ACF my_acfs.hs my_atten_image.hv Siemens_mMR_seg2.hs > my_create_acfs.log 2>&1
if [ $? -ne 0 ]; then
echo "ERROR running calculate_attenuation_coefficients. Check my_create_acfs.log"; exit 1;
fi

echo "=== Creating my_test_lm_frame.fdef (time frame definitions)"
# Note: test data contains only 612 ms of data, so use a very short frame of 0.5s
rm -f my_test_lm_frame.fdef
echo "0 0.1" > my_test_lm_frame.fdef # skip the first .1s, to test if this feature works
echo "1 0.5" >> my_test_lm_frame.fdef
export FRAMES

for use_frame in true false; do
    if $use_frame; then
        FRAMES=my_test_lm_frame.fdef
        suffix=frame
    else
        FRAMES=""
        suffix=counts
    fi
    echo "=============== Using ${suffix} definition ============="

    echo "=== Reconstruct listmode data without cache"
    export filename=my_output_t_lm_pr_seg2_${suffix}
    export cache=0
    export recompute_cache=0
    logfile=OSMAPOSL_test_lm_${suffix}_1.log
    if ${MPIRUN} OSMAPOSL OSMAPOSL_test_lm.par > "$logfile" 2>&1
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
    export filename=my_output_t_lm_pr_seg2_${suffix}_with_new_cache
    export cache=5000
    export recompute_cache=1
    logfile=OSMAPOSL_test_lm_${suffix}_2.log
    if ${MPIRUN} OSMAPOSL OSMAPOSL_test_lm.par > "$logfile" 2>&1
    then
        echo "---- Executable ran ok"
    else
        echo "---- There were problems here!"
        ThereWereErrors=1;
        ErrorLogs="$ErrorLogs $logfile"
    fi

    echo "=== Compare reconstructed images with and without caching LM file"
    logfile=my_output_comparison_nocache_vs_new_cache_${suffix}.log
    if compare_image my_output_t_lm_pr_seg2_${suffix}_1.hv my_output_t_lm_pr_seg2_${suffix}_with_new_cache_1.hv > "$logfile" 2>&1
    then
        echo "---- This test seems to be ok !"
    else
        echo "---- There were problems here!"
        ThereWereErrors=1;
        ErrorLogs="$ErrorLogs $logfile"
    fi

    echo "=== Reconstruct listmode data with cache loaded from the disk"
    export filename=my_output_t_lm_pr_seg2_${suffix}_with_old_cache
    export cache=40000
    export recompute_cache=0
    logfile=OSMAPOSL_test_lm_${suffix}_3.log
    if ${MPIRUN} OSMAPOSL OSMAPOSL_test_lm.par > "$logfile" 2>&1
    then
        echo "---- Executable ran ok"
    else
        echo "---- There were problems here!"
        ThereWereErrors=1;
        ErrorLogs="$ErrorLogs $logfile"
    fi

    echo "=== Compare reconstructed images without caching LM file and with loading cache from disk"
    logfile=my_output_comparison_nocache_vs_existing_cache_${suffix}.log
    if compare_image my_output_t_lm_pr_seg2_${suffix}_1.hv my_output_t_lm_pr_seg2_${suffix}_with_old_cache_1.hv > "$logfile" 2>&1
    then
        echo "---- This test seems to be ok !"
    else
        echo "---- There were problems here!"
        ThereWereErrors=1;
        ErrorLogs="$ErrorLogs $logfile"
    fi

    # create sinograms
    echo "=== Unlist listmode data (for comparison)"
    logfile=lm_to_projdata_${suffix}.log
    if env INPUT=PET_ACQ_small.l.hdr.STIR TEMPLATE=Siemens_mMR_seg2.hs OUT_PROJDATA_FILE=my_sinogram lm_to_projdata  lm_to_projdata.par > "$logfile" 2>&1
    then
        echo "---- Executable ran ok"
    else
        echo "---- There were problems here!"
        ThereWereErrors=1;
        ErrorLogs="$ErrorLogs $logfile"
    fi

    echo "=== Reconstruct projection data for comparison"
    export filename=my_output_t_proj_seg2_${suffix}
    logfile=OSMAPOSL_test_proj_${suffix}.log
    if ${MPIRUN} OSMAPOSL OSMAPOSL_test_proj.par > "$logfile" 2>&1
    then
        echo "---- Executable ran ok"
    else
        echo "---- There were problems here!"
        ThereWereErrors=1;
        ErrorLogs="$ErrorLogs $logfile"
    fi

    echo "=== Compare sensitivity images"
    logfile=my_sens_comparison_${suffix}.log
    if compare_image my_sens_t_proj_seg2.hv my_sens_t_lm_pr_seg2.hv > "$logfile" 2>&1
    then
        echo "---- This test seems to be ok !"
    else
        echo "---- There were problems here!"
        ThereWereErrors=1;
        ErrorLogs="$ErrorLogs $logfile"
    fi

    echo "=== Compare reconstructed images"
    logfile=my_output_comparison_proj_vs_lm_${suffix}.log
    if compare_image my_output_t_proj_seg2_${suffix}_1.hv my_output_t_lm_pr_seg2_${suffix}_1.hv > "$logfile" 2>&1
    then
        echo "---- This test seems to be ok !"
    else
        echo "---- There were problems here!"
        ThereWereErrors=1;
        ErrorLogs="$ErrorLogs $logfile"
    fi
done

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
