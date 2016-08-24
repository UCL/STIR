#! /bin/sh
# A script to check to see if reconstruction of simulated data gives the expected result.
#
#  Copyright (C) 2011 - 2011-01-14, Hammersmith Imanet Ltd
#  Copyright (C) 2011-07-01 - 2011, Kris Thielemans
#  Copyright (C) 2014, University College London
#  This file is part of STIR.
#
#  This file is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 2.1 of the License, or
#  (at your option) any later version.

#  This file is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  See STIR/LICENSE.txt for details
#      
# Author Kris Thielemans
# 

# Scripts should exit with error code when a test fails:
if [ -n "$TRAVIS" ]; then
    # The code runs inside Travis
    set -e
fi

echo This script should work with STIR version 2.2, 2.3, 2.4 and 3.0. If you have
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

command -v generate_image >/dev/null 2>&1 || { echo "generate_image not found or not executable. Aborting." >&2; exit 1; }
echo "Using `command -v generate_image`"
echo "Using `command -v OSMAPOSL`"

# first need to set this to the C locale, as this is what the STIR utilities use
# otherwise, awk might interpret floating point numbers incorrectly
LC_ALL=C
export LC_ALL

echo "===  make emission image"
generate_image  generate_uniform_cylinder.par
echo "===  use that as template for attenuation"
stir_math --including-first --times-scalar .096 my_atten_image.hv my_uniform_cylinder.hv
echo "===  create template sinogram (DSTE in 3D with max ring diff 2 to save time)"
template_sino=my_DSTE_3D_rd2_template.hs
cat > my_input.txt <<EOF
Discovery STE
1
n

0
2
EOF
create_projdata_template  ${template_sino} < my_input.txt > my_create_${template_sino}.log 2>&1
if [ $? -ne 0 ]; then 
  echo "ERROR running create_projdata_template. Check my_create_${template_sino}.log"; exit 1; 
fi

# create sinograms
./simulate_data.sh my_uniform_cylinder.hv my_atten_image.hv ${template_sino}
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
for recon in FBP2D FBP3DRP OSMAPOSL OSSPS; do
  echo "Using `command -v ${recon}`"
  for parfile in ${recon}_test_sim*.par; do
    echo "============================================="
    # test first if analytic reconstruction and if so, run pre-correction
    isFBP=0
    if expr ${recon} : FBP > /dev/null; then
      isFBP=1
      echo "Running precorrection"
      correct_projdata correct_projdata_simulation.par > my_correct_projdata_simulation.log 2>&1
      if [ $? -ne 0 ]; then
        echo "Error running precorrection. CHECK my_correct_projdata_simulation.log"
        error_log_files="${error_log_files} my_correct_projdata_simulation.log"
        break
      fi
    fi

    # run actual reconstruction
    echo "Running ${recon} ${parfile}"
    ${MPIRUN} ${recon} ${parfile} > my_${parfile}.log 2>&1
    if [ $? -ne 0 ]; then
       echo "Error running reconstruction. CHECK RECONSTRUCTION LOG my_${parfile}.log"
       error_log_files="${error_log_files} my_${parfile}.log"
       break
    fi

    # find filename of (last) image from ${parfile}
    output_filename=`awk -F':='  '/output[ _]*filename[ _]*prefix/ { value=$2;gsub(/[ \t]/, "", value); printf("%s", value) }' ${parfile}`
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
      echo "DIFFERENCE IN ROI VALUES IS TOO LARGE. CHECK RECONSTRUCTION LOG my_${parfile}.log"
      error_log_files="${error_log_files} my_${parfile}.log"
    else
      echo "This seems fine."
    fi

    echo "============================================="
  done
done

if [ -z "${error_log_files}" ]; then
 echo "All tests OK!"
 echo "You can remove all output using \"rm -f my_*\""
else
 echo "There were errors. Check ${error_log_files}"
fi

