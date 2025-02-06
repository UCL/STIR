#! /bin/sh
# A script to check to see if reconstruction of simulated data gives the expected result.
#
#  Copyright (C) 2011 - 2011-01-14, Hammersmith Imanet Ltd
#  Copyright (C) 2011-07-01 - 2011, Kris Thielemans
#  Copyright (C) 2014, 2022, 2024 University College London
#  This file is part of STIR.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  See STIR/LICENSE.txt for details
#       
# Author Kris Thielemans
# Author Dimitra Kyriakopoulou

echo This script should work with STIR version 6.2. If you have
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
echo "Using `command -v OSSPS`"
echo "Using `command -v FBP2D`"
echo "Using `command -v FBP3DRP`"
echo "Using `command -v SRT2D`"
echo "Using `command -v SRT2DSPECT`"

# first need to set this to the C locale, as this is what the STIR utilities use
# otherwise, awk might interpret floating point numbers incorrectly
LC_ALL=C
export LC_ALL

./simulate_data_for_tests.sh
if [ $? -ne 0 ]; then
  echo "Error running simulation"
  exit 1
fi
# need to repeat with zero-offset now as FBP doesn't support it
zero_view_suffix=_force_zero_view_offset
./simulate_data_for_tests.sh --force_zero_view_offset --suffix $zero_view_suffix
if [ $? -ne 0 ]; then
  echo "Error running simulation with zero view offset"
  exit 1
fi
## TOF data
TOF_suffix=_TOF
./simulate_data_for_tests.sh --TOF --suffix "$TOF_suffix"
if [ $? -ne 0 ]; then
  echo "Error running simulation"
  exit 1
fi
## SPECT data
SPECT_suffix=_SPECT 
./simulate_data_for_tests.sh --SPECT --suffix "$SPECT_suffix"
if [ $? -ne 0 ]; then
  echo "Error running simulation"
  exit 1
fi

error_log_files=""

input_image=my_uniform_cylinder.hv
input_voxel_size_x=`stir_print_voxel_sizes.sh ${input_image}|awk '{print $3}'`
ROI=ROI_uniform_cylinder.par
list_ROI_values ${input_image}.roistats ${input_image} ${ROI} 0 > /dev/null 2>&1
input_ROI_mean=`awk 'NR>2 {print $2}' ${input_image}.roistats`

# loop over reconstruction algorithms
# warning: currently OSMAPOSL needs to be run before OSSPS as 
# the OSSPS par file uses an OSMAPOSL result as initial image
# and reuses its subset sensitivities
for recon in FBP2D FBP3DRP SRT2D SRT2DSPECT OSMAPOSL OSSPS ; do  
  echo "========== Testing `command -v ${recon}`"
  # Check if we have CUDA code and parallelproj.
  # If so, check for test files in CUDA/*
  if stir_list_registries |grep -i cuda > /dev/null
  then
      if stir_list_registries |grep -i parallelproj > /dev/null
      then
          extra_par_files=`ls CUDA/${recon}_test_sim*.par 2> /dev/null`
          if [ -n "$TRAVIS" -o -n "$GITHUB_WORKSPACE" ]; then
              # The code runs inside Travis or GHA
              if [ -n "$extra_par_files" ]; then
                  echo "Not running ${extra_par_files} due to no CUDA run-time"
                  extra_par_files=""
              fi
          fi
      fi
  fi
  for parfile in ${recon}_test_sim*.par ${extra_par_files}; do
    for dataSuffix in "" "$TOF_suffix"; do
      echo "===== data suffix: \"$dataSuffix\""
      # test first if analytic reconstruction and if so, run pre-correction
      is_analytic=0
      if expr "$recon" : FBP > /dev/null; then
        is_analytic=1
      elif expr "$recon" : SRT > /dev/null; then
        is_analytic=1
      fi
      if [ $is_analytic = 1 ]; then
          if expr "$dataSuffix" : '.*TOF.*' > /dev/null; then
            echo "Skipping TOF as not yet supported for FBP and SRT"
            break
          fi
	  if expr "$recon" : SRT2DSPECT > /dev/null; then
	    suffix=$SPECT_suffix
	    export suffix
          else   
            suffix=$zero_view_suffix
            export suffix
            echo "Running precorrection"
	    correct_projdata correct_projdata_simulation.par > my_correct_projdata_simulation.log 2>&1
	    if [ $? -ne 0 ]; then
              echo "Error running precorrection. CHECK my_correct_projdata_simulation.log"
	      error_log_files="${error_log_files} my_correct_projdata_simulation.log"
	      break
	    fi
          fi
      else
          suffix="$dataSuffix"
          export suffix
          # we simulate 2 different scanners for non-TOF and TOF that sadly need different number of subsets
          if expr "$dataSuffix" : '.*TOF.*' > /dev/null; then
              num_subsets=12
          else
              num_subsets=14
          fi
          export num_subsets
      fi

      # run actual reconstruction
      echo "Running ${recon} ${parfile}"
      logfile="my_`basename ${parfile}`${suffix}.log"
      ${MPIRUN} ${recon} ${parfile} > "$logfile" 2>&1
      if [ $? -ne 0 ]; then
          echo "Error running reconstruction. CHECK RECONSTRUCTION LOG \"$logfile\""
          error_log_files="${error_log_files} "$logfile""
          break
      fi

      # find filename of (last) image from ${parfile}
      output_filename=`awk -F':='  '/output[ _]*filename[ _]*prefix/ { value=$2;gsub(/[ \t]/, "", value); printf("%s", value) }' "$parfile"`
      # substitute env variables (e.g. to fill in suffix)
      output_filename=`eval echo "${output_filename}"`
      if [ ${is_analytic} -eq 0 ]; then
          # iterative algorithm, so we need to append the num_subiterations
          num_subiterations=`awk -F':='  '/number[ _]*of[ _]*subiterations/ { value=$2;gsub(/[ \t]/, "", value); printf("%s", value) }' ${parfile}`
          output_filename=${output_filename}_${num_subiterations}
      fi
      output_image=${output_filename}.hv

      # compute ROI value
      list_ROI_values ${output_image}.roistats ${output_image} ${ROI} 0  > ${output_image}.roistats.log 2>&1
      if [ $? -ne 0 ]; then
          echo "Error running list_ROI_values. CHECK LOG ${output_image}.roistats.log"
          error_log_files="${error_log_files} ${output_image}.roistats.log"
          break
      fi

      # compare ROI value
      output_voxel_size_x=`stir_print_voxel_sizes.sh ${output_image}|awk '{print $3}'`
      output_ROI_mean=`awk "NR>2 {print \\$2*${input_voxel_size_x}/${output_voxel_size_x}}" ${output_image}.roistats`
      echo "Input ROI mean: $input_ROI_mean"
      echo "Output ROI mean: $output_ROI_mean"
      error_bigger_than_1percent=`echo $input_ROI_mean $output_ROI_mean| awk '{ print(($2/$1 - 1)*($2/$1 - 1)>0.0001) }'`
      if [ ${error_bigger_than_1percent} -eq 1 ]; then
          echo "DIFFERENCE IN ROI VALUES IS TOO LARGE. CHECK RECONSTRUCTION LOG "$logfile""
          error_log_files="${error_log_files} ${logfile}"
      else
          echo "This seems fine."
      fi

    done
  done
done

echo "============================================="
if [ -z "${error_log_files}" ]; then
 echo "All tests OK!"
 echo "You can remove all output using \"rm -f my_*\""
 exit 0
else
 echo "There were errors. Check ${error_log_files}"
 tail ${error_log_files}
 exit 1
fi
