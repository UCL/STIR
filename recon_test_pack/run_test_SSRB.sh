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
# 

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

echo "Using `command -v SSRB`"
echo "Using `command -v OSMAPOSL`"

# first need to set this to the C locale, as this is what the STIR utilities use
# otherwise, awk might interpret floating point numbers incorrectly
LC_ALL=C
export LC_ALL

echo "=== Creating single view data and test TOF rebinning ==="
logfile=SSRB_sim1.1og
max_rd=9 span=3 view_mash=288 tof_mash=1 ./simulate_PET_data_for_tests.sh --suffix _TOF_vm288_rd9_span3 --TOF > "$logfile" 2>&1
if [ $? -ne 0 ]; then
  echo "Error running simulation, Check $logfile"
  exit 1
fi
logfile=SSRB_sim2.1og
max_rd=9 span=3 view_mash=288 tof_mash=11 ./simulate_PET_data_for_tests.sh --keep_images --suffix _TOF_vm288_rd9_span3_tm11 --TOF > "$logfile" 2>&1
if [ $? -ne 0 ]; then
  echo "Error running simulation, Check $logfile"
  exit 1
fi
logfile=SSRB_run_SSRB_sim1.log
SSRB --template my_line_integrals_TOF_vm288_rd9_span3_tm11.hs my_line_integrals_TOF_vm288_rd9_span3_TOFSSRB11.hs my_line_integrals_TOF_vm288_rd9_span3.hs 0 > "$logfile" 2>&1
if [ $? -ne 0 ]; then
  echo "Error running SSRB, Check $logfile"
  exit 1
fi
compare_projdata my_line_integrals_TOF_vm288_rd9_span3_tm11.hs  my_line_integrals_TOF_vm288_rd9_span3_TOFSSRB11.hs
if [ $? -ne 0 ]; then
  echo "Error comparing TOF mash simulation with TOF-SSRB"
  exit 1
fi


echo ""
echo "=== Creating 3D data and test axial+TOF rebinning via reconstruction ==="
echo "We will simulate some data at span=1 and TOF-mashing=1, rebin it, then reconstruct."
echo "The image will be checked by using an ROI calculation."

echo "Running simulation (will take a while)"
max_rd=5
export max_rd
view_mash=8
export view_mash
suffix="_TOF_vm${view_mash}_rd${max_rd}_span1"
logfile=SSRB_sim3.1og
span=1 tof_mash=1 background_value=.1 ./simulate_PET_data_for_tests.sh --keep_images --suffix "$suffix" --TOF > "$logfile" 2>&1
if [ $? -ne 0 ]; then
  echo "Error running simulation, Check $logfile"
  exit 1
fi
echo "Rebinning data"
tof_mash=11
span=3
suffixSSRB="_TOF_vm${view_mash}_rd${max_rd}_span${span}_TOFSSRB${tof_mash}"
logfile="SSRB_run_SSRB_line_integrals$suffixSSRB.1og"
SSRB "my_line_integrals${suffixSSRB}.hs" "my_line_integrals${suffix}.hs" "$span" 1 1 "$max_rd" "$tof_mash" > "$logfile" 2>&1
if [ $? -ne 0 ]; then
  echo "Error running SSRB, Check $logfile"
  exit 1
fi
logfile="SSRB_run_SSRB_acfs$suffixSSRB.1og"
SSRB "my_acfs${suffixSSRB}.hs" "my_acfs${suffix}.hs" "$span" 1 1 "$max_rd" "$tof_mash" > "$logfile" 2>&1
if [ $? -ne 0 ]; then
  echo "Error running SSRB, Check $logfile"
  exit 1
fi
logfile="SSRB_run_SSRB_prompts$suffixSSRB.1og"
SSRB "my_prompts${suffixSSRB}.hs" "my_prompts${suffix}.hs" "$span" 1 1 "$max_rd" "$tof_mash" > "$logfile" 2>&1
if [ $? -ne 0 ]; then
  echo "Error running SSRB, Check $logfile"
  exit 1
fi
logfile="SSRB_run_SSRB_additive_sinograms$suffixSSRB.1og"
SSRB "my_additive_sinogram${suffixSSRB}.hs" "my_additive_sinogram${suffix}.hs" "$span" 1 1 "$max_rd" "$tof_mash" > "$logfile" 2>&1
if [ $? -ne 0 ]; then
  echo "Error running SSRB, Check $logfile"
  exit 1
fi


# Below code is a copy of relevant lines in run_test_simulate_and_recon.sh.
# It would be far better to make those into functions/scripts...

recon=OSMAPOSL
isFBP=0
parfile=${recon}_test_sim_PM.par
suffix=${suffixSSRB}
export suffix

num_subsets=2
export num_subsets

input_image=my_uniform_cylinder.hv
input_voxel_size_x=`stir_print_voxel_sizes.sh ${input_image}|awk '{print $3}'`
ROI=ROI_uniform_cylinder.par
list_ROI_values ${input_image}.roistats ${input_image} ${ROI} 0 > /dev/null 2>&1
input_ROI_mean=`awk 'NR>2 {print $2}' ${input_image}.roistats`

# Indentation is here because straight copy of the file.

      # run actual reconstruction
      echo "Running ${recon} ${parfile}"
      logfile="my_${parfile}${suffix}.log"
      ${MPIRUN} ${recon} ${parfile} > "$logfile" 2>&1
      if [ $? -ne 0 ]; then
          echo "Error running reconstruction. CHECK RECONSTRUCTION LOG \"$logfile\""
          exit 1
      fi

      # find filename of (last) image from ${parfile}
      output_filename=`awk -F':='  '/output[ _]*filename[ _]*prefix/ { value=$2;gsub(/[ \t]/, "", value); printf("%s", value) }' "$parfile"`
      # substitute env variables (e.g. to fill in suffix)
      output_filename=`eval echo "${output_filename}"`
      if [ ${isFBP} -eq 0 ]; then
          # iterative algorithm, so we need to append the num_subiterations
          num_subiterations=`awk -F':='  '/number[ _]*of[ _]*subiterations/ { value=$2;gsub(/[ \t]/, "", value); printf("%s", value) }' ${parfile}`
          output_filename=${output_filename}_${num_subiterations}
      fi
      output_image=${output_filename}.hv

      # compute ROI value
      list_ROI_values ${output_image}.roistats ${output_image} ${ROI} 0  > ${output_image}.roistats.log 2>&1
      if [ $? -ne 0 ]; then
          echo "Error running list_ROI_values. CHECK LOG ${output_image}.roistats.log"
          exit 1
      fi

      # compare ROI value
      output_voxel_size_x=`stir_print_voxel_sizes.sh ${output_image}|awk '{print $3}'`
      output_ROI_mean=`awk "NR>2 {print \\$2*${input_voxel_size_x}/${output_voxel_size_x}}" ${output_image}.roistats`
      echo "Input ROI mean: $input_ROI_mean"
      echo "Output ROI mean: $output_ROI_mean"
      error_bigger_than_1percent=`echo $input_ROI_mean $output_ROI_mean| awk '{ print(($2/$1 - 1)*($2/$1 - 1)>0.0001) }'`
      if [ ${error_bigger_than_1percent} -eq 1 ]; then
          echo "DIFFERENCE IN ROI VALUES IS TOO LARGE. CHECK RECONSTRUCTION LOG "$logfile""
      else
          echo "This seems fine."
      fi

