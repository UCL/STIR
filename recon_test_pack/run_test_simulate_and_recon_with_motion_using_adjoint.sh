#! /bin/sh
# A script to check to see if reconstruction of simulated data gives the expected result.
#
#  Copyright (C) 2011 - 2011-01-14, Hammersmith Imanet Ltd
#  Copyright (C) 2011-07-01 - 2011, Kris Thielemans
#  Copyright (C) 2014 - 2020, University College London
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
# Author Richard Brown
# 

# Scripts should exit with error code when a test fails:
if [ -n "$TRAVIS" ]; then
    # The code runs inside Travis
    set -e
fi

echo This script should work with STIR version 4.0. If you have
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
echo "===  make attenuation image"
generate_image  generate_atten_cylinder.par
input_image=my_uniform_cylinder.hv
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
./simulate_data.sh ${input_image} my_atten_image.hv ${template_sino}
if [ $? -ne 0 ]; then
  echo "Error running simulation"
  exit 1
fi
cat > my_multi_sinos.txt <<EOF
Multi :=
  total number of data sets := 1
  data set[1] := my_prompts.hs
end :=
EOF

error_log_files=""

input_voxel_size_x=`stir_print_voxel_sizes.sh ${input_image}|awk '{print $3}'`
ROI=ROI_uniform_cylinder.par
list_ROI_values ${input_image}.roistats ${input_image} ${ROI} 0 > /dev/null 2>&1
input_ROI_mean=`awk 'NR>2 {print $2}' ${input_image}.roistats`

echo "===  make displacement field images"
motion_image_3d=my_uniform_cylinder_motion_3d.hv
stir_math --times-scalar 0 $motion_image_3d $input_image


par_file=OSMAPOSL_with_motion_test_using_adjoint.par
log_file=my_rec_OSMAPOSL_with_motion_test_using_adjoint.log
echo "Running OSMAPOSL $par_file for 1 iteration (?? subsets)"
${MPIRUN} OSMAPOSL $par_file #> $log_file 2>&1
if [ $? -ne 0 ]; then
   echo "Error running reconstruction. CHECK RECONSTRUCTION LOG ${log_file}"
   error_log_files="${error_log_files} ${log_file}"
   exit 1
fi
output_image=my_osmaposl_with_motion_using_adjoint_1.hv
compare_image -t 0.45 $output_image ${input_image}
if [ $? -ne 0 ]; then
  echo "ERROR comparison of reconstructed image fails"; exit 1;
fi

if [ -z "${error_log_files}" ]; then
 echo "All tests OK!"
 echo "You can remove all output using \"rm -f my_*\""
 exit 0
else
 echo "There were errors. Check ${error_log_files}"
 tail ${error_log_files}
 exit 1
fi

