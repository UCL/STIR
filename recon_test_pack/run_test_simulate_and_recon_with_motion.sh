#! /bin/sh
# A script to check to see if warping and motion incorporated reconstruction of simulated data gives the expected result.
#
#  Copyright (C) 2011 - 2011-01-14, Hammersmith Imanet Ltd
#  Copyright (C) 2011 Kris Thielemans
#  Copyright (C) 2013 King's College London
#  Copyright (C) 2013 - 2014, University College London
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
# Author Charalampos Tsoumpas
# Author Kris Thielemans
# 

# Scripts should exit with error code when a test fails:
if [ -n "$TRAVIS" ]; then
    # The code runs inside Travis
    set -e
fi

echo This script should work with STIR version 2.4 and 3.0. If you have
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
command -v warp_and_accumulate_gated_images >/dev/null 2>&1 || { echo "warp_and_accumulate_gated_images not found or not executable. Aborting." >&2; exit 1; }
echo "Using `command -v warp_and_accumulate_gated_images`"
echo "Using `command -v OSMAPOSL`"

# first need to set this to the C locale, as this is what the STIR utilities use
# otherwise, awk might interpret floating point numbers incorrectly
LC_ALL=C
export LC_ALL

echo "===  make emission image"
generate_image  generate_test_object.par
input_image=my_test_object_g1.hv
input_voxel_size_x=`stir_print_voxel_sizes.sh ${input_image}|awk '{print $3}'`
ROI=ROI_uniform_cylinder.par
list_ROI_values ${input_image}.roistats ${input_image} ${ROI} 0 > /dev/null 2>&1
input_ROI_mean=`awk 'NR>2 {print $2}' ${input_image}.roistats`

echo "===  use that as template for attenuation"
stir_math --including-first --times-scalar .096 my_att_test_object_g1 my_test_object_g1.hv
convert_to_binary_image my_uniform_image_all1 my_test_object_g1.hv -1
stir_math --including-first --times-scalar 3.27 my_translation_g2d1.hv my_uniform_image_all1.hv  #z axis shift: 1*z_axis_voxel_size
stir_math --including-first --times-scalar 4.794 my_translation_g2d2.hv my_uniform_image_all1.hv  #y axis shift: 2*y_axis_voxel_size
stir_math --including-first --times-scalar 7.191 my_translation_g2d3.hv my_uniform_image_all1.hv  #x axis shift: 3*x_axis_voxel_size
stir_math --including-first --times-scalar -3.27 my_reverse_translation_g2d1.hv my_uniform_image_all1.hv  #z axis reverse shift: -1*z_axis_voxel_size
stir_math --including-first --times-scalar -4.794 my_reverse_translation_g2d2.hv my_uniform_image_all1.hv  #y axis reverse shift: -2*y_axis_voxel_size
stir_math --including-first --times-scalar -7.191 my_reverse_translation_g2d3.hv my_uniform_image_all1.hv  #x axis reverse shift: -3*x_axis_voxel_size
echo "===  warp image"
warp_image my_test_object_g2 my_test_object_g1.hv my_translation_g2d3.hv my_translation_g2d2.hv my_translation_g2d1.hv 1 0
stir_math --including-first --times-scalar .096 my_att_test_object_g2 my_test_object_g2.hv
for d in 1 2 3 ; do 
convert_to_binary_image my_reverse_translation_g1d${d}.hv my_test_object_g1.hv 2
convert_to_binary_image my_translation_g1d${d}.hv my_test_object_g1.hv 2
done
cat > my_template.gdef <<EOF
1 1
2 1
EOF
cp my_template.gdef my_translation.gdef
cp my_template.gdef my_reverse_translation.gdef
cp my_template.gdef my_test_object.gdef
echo "===  warp back the shifted test_object and the original one"
warp_and_accumulate_gated_images my_twice_test_object my_test_object my_reverse_translation
stir_math --including-first --times-scalar 2 my_2times_test_object.hv my_test_object_g1.hv
compare_image -t 0.0001 my_2times_test_object.hv my_twice_test_object.hv
if [ $? -ne 0 ]; then
  echo "ERROR running warp_and_accumulate_gated_images"; exit 1;
fi

echo "===  create template sinogram (DSTE in 3D with max ring diff 2 to save time)"
template_sino=my_DSTE_3D_rd2_template.hs
cat > my_input.txt <<EOF
Discovery STE
-1
-1
1
n

0
2
EOF
create_projdata_template  ${template_sino} < my_input.txt > my_create_${template_sino}.log 2>&1
if [ $? -ne 0 ]; then 
  echo "ERROR running create_projdata_template. Check my_create_${template_sino}.log"; exit 1; 
fi

for g in 1 2 ; do 
echo "===  creating ACF for gate $g"
calculate_attenuation_coefficients --ACF my_ACF_test_object_g$g my_att_test_object_g$g.hv ${template_sino} > my_create_ACF_g$g.log 2>&1
echo "===  creating sinogram for gate $g"
forward_project my_fwd_test_object_g$g  my_test_object_g$g.hv ${template_sino} forward_projector_proj_matrix_ray_tracing.par > my_fwd_test_object_g$g.log 2>&1
stir_math -s --mult --power -1 my_att_fwd_test_object_g$g  my_fwd_test_object_g$g.hs my_ACF_test_object_g$g.hs
done

if [ $? -ne 0 ]; then
  echo "Error running simulation"
  exit 1
fi

error_log_files=""
cp my_template.gdef my_att_fwd_test_object.gdef
cp my_template.gdef my_ACF_test_object.gdef

    echo "Running OSMAPOSL OSMAPOSL_with_motion_test.par for 1 iteration (28 subsets)"
    ${MPIRUN} OSMAPOSL OSMAPOSL_with_motion_test.par > my_rec_test_object.log 2>&1
    if [ $? -ne 0 ]; then
       echo "Error running reconstruction. CHECK RECONSTRUCTION LOG my_rec_test_object.log"
       error_log_files="${error_log_files} my_rec_test_object.log"
       exit 1
    fi
    compare_image -t 0.25 my_rec_test_object_46.hv my_test_object_g1.hv
if [ $? -ne 0 ]; then
  echo "ERROR comparison of reconstructed image fails"; exit 1;
fi
    output_image=my_rec_test_object_46.hv
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
      echo "DIFFERENCE IN ROI VALUES IS TOO LARGE. CHECK RECONSTRUCTION LOG ${parfile}.log"
      error_log_files="${error_log_files} ${parfile}.log"
    else
      echo "This seems fine."
    fi

if [ -z "${error_log_files}" ]; then
 echo "All tests OK!"
 echo "You can remove all output using \"rm -f my_*\""
else
 echo "There were errors. Check ${error_log_files}"
fi
