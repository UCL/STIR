#! /bin/sh
# A script to check to see if zoom_image scales things ok.
#
#  Copyright (C) 2005, Hammersmith Imanet Ltd (boiler plate code for option processing)
#  Copyright (C) 2019, University College London
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

echo This script should work with STIR version 4.0. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.
echo

#
# Parse option arguments (--)
# Note that the -- is required to suppress interpretation of $1 as options 
# to expr
#
while test `expr -- "$1" : "--.*"` -gt 0
do

  if test "$1" = "--help"
  then
    echo "Usage: `basename $0` [install_dir]"
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
echo "Using `command -v zoom_image`"

# first need to set this to the C locale, as this is what the STIR utilities use
# otherwise, awk might interpret floating point numbers incorrectly
LC_ALL=C
export LC_ALL

# define ROI function for convenience (image as argument, prints value to stdout)
get_ROI_value() {
  list_ROI_values $1.roistats $1 ${ROI} 0 > $1.roistats.log 2>&1
  if [ $? -ne 0 ]; then
    echo "Error running list_ROI_values. CHECK LOG ${output_image}.roistats.log"
    exit 1
  fi
  awk 'NR>2 {print $2}' $1.roistats
}

# function to compare values
# compare_values x y max_relative_error
# warning: return 0 if they are different (which is non-standard I guess)
compare_values() {
    if [ $# -ne 3 ]; then
      echo "Something wrong with call to compare_values"
    exit 1
    fi
    error_bigger_than_x=`echo $1 $2 $3 | awk '{ print(($2/$1 - 1)*($2/$1 - 1)> ($3 * $3)) }'`
  if [ $error_bigger_than_x -eq 1 ]; then
    echo "Difference between $1 and $2 too large"
    return 0
  else
    return 1
  fi
}

echo "===  make emission image"
generate_image  generate_uniform_cylinder.par

error_log_files=""

input_image=my_uniform_cylinder.hv
ROI=ROI_uniform_cylinder.par
org_ROI_mean=`get_ROI_value ${input_image}`

echo "===  running zoom_image and compare ROI values"

# running with same template (no changes expected)
zoom_image --template $input_image my_zoom_test.hv $input_image
ROI_mean=`get_ROI_value  my_zoom_test.hv`
if compare_values $org_ROI_mean $ROI_mean .01
then
  echo "DIFFERENCE IN ROI VALUES after zoom_image with template IS TOO LARGE."
  exit 1
fi

# running with zoom
zoom=2.1
zoom_image my_zoom_test2.hv $input_image 201 $zoom
ROI_mean=`get_ROI_value  my_zoom_test2.hv`
scaled_ROI_mean=`echo $ROI_mean $zoom| awk '{ print $1*$2*$2 }'`
if compare_values $org_ROI_mean $scaled_ROI_mean .01
then
  echo "DIFFERENCE IN ROI VALUES after zoom_image with preserve_sum IS TOO LARGE."
  exit 1
fi

zoom=2.1
zoom_image  --scaling preserve_values my_zoom_test3.hv $input_image 201 $zoom
ROI_mean=`get_ROI_value  my_zoom_test3.hv`
if compare_values $org_ROI_mean $ROI_mean .01
then
  echo "DIFFERENCE IN ROI VALUES after zoom_image with preserve_sum IS TOO LARGE."
  exit 1
fi

# test for forward projection

zoom=2.1
zoom_image  --scaling preserve_projections my_zoom_test4.hv $input_image 201 $zoom
ROI_mean=`get_ROI_value  my_zoom_test4.hv`
scaled_ROI_mean=`echo $ROI_mean $zoom| awk '{ print $1*$2 }'`
if compare_values $org_ROI_mean $scaled_ROI_mean .01
then
  echo "DIFFERENCE IN ROI VALUES after zoom_image with preserve_sum IS TOO LARGE."
  exit 1
fi

echo "===  make attenuation image"
generate_image  generate_atten_cylinder.par

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

echo "===  running projections to check if result is the same when using preserve-projections"

# create sinograms (use zero background)
./simulate_data.sh my_uniform_cylinder.hv my_atten_image.hv ${template_sino} 0
if [ $? -ne 0 ]; then
  echo "Error running simulation"
  exit 1
fi

org_sum=`list_projdata_info --sum my_prompts.hs | awk -F: '{ print $2}'`

./simulate_data.sh my_zoom_test4.hv my_atten_image.hv ${template_sino} 0
if [ $? -ne 0 ]; then
  echo "Error running simulation"
  exit 1
fi

new_sum=`list_projdata_info --sum my_prompts.hs | awk -F: '{ print $2}'`

if compare_values $org_sum $new_sum .01
then
  echo "DIFFERENCE IN su  of prompts IS TOO LARGE."
  exit 1
fi

if [ -z "${error_log_files}" ]; then
 echo "All tests OK!"
 echo "You can remove all output using \"rm -f my_*\""
else
 echo "There were errors. Check ${error_log_files}"
fi

