#! /bin/sh -vx
# A script to simulate some data used by other tests.
# Careful: run_scatter_tests.sh compares with previously generated data
# so you cannot simply modify this one without adjusting that as well.
#
# This script is not intended to be run on its own!
#
#  Copyright (C) 2011 - 2011-01-14, Hammersmith Imanet Ltd
#  Copyright (C) 2011-07-01 - 2011, Kris Thielemans
#  Copyright (C) 2014, 2020, 2022, 2024 University College London
#  This file is part of STIR.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  See STIR/LICENSE.txt for details
#      
# Author Kris Thielemans
# Author Dimitra Kyriakopoulou 

echo This script should work with STIR version 6.x. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.
echo

command -v generate_image >/dev/null 2>&1 || { echo "generate_image not found or not executable. Aborting." >&2; exit 1; }
echo "Using `command -v generate_image`"

generate_images=1
force_zero_view_offset=0
TOF=0
suffix=""
SPECT=0
#
# Parse option arguments (--)
# Note that the -- is required to suppress interpretation of $1 as options 
# to expr
#
while test `expr -- "$1" : "--.*"` -gt 0
do

  if test "$1" = "--force_zero_view_offset"
  then
    force_zero_view_offset=1
  elif test "$1" = "--TOF"
  then
    TOF=1
  elif test "$1" = "--SPECT"
  then
    SPECT=1
  elif test "$1" = "--suffix"
  then
    suffix="$2"
    shift 1
  elif test "$1" = "--keep_images"
  then
    generate_images=0
  elif test "$1" = "--help"
  then
    echo "Usage: `basename $0` [--force_zero_view_offset]  [--suffix sometext] [--keep_images] [install_dir]"
    echo "(where [] means that an argument is optional)"
    echo "The suffix is appended to filenames (before adding the extensions .hs/.s)."
    echo "--keep-images can be used to avoid regenerating the images (if you know they are correct)."
    echo "You can choose data sizes different from the defaults by setting the following environment variables:"
    echo "   view_mash, span, max_rd, tof_mash"
    echo "Randoms+scatter are set to a constant proj_data. you can set that value via the environment variable:"
    echo "   background_value"
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

# first need to set this to the C locale, as this is what the STIR utilities use
# otherwise, awk might interpret floating point numbers incorrectly
LC_ALL=C
export LC_ALL

if [ "$generate_images" -eq 1 ]; then
  echo "===  make emission image"
  generate_image  generate_uniform_cylinder.par
  echo "===  make attenuation image"
  generate_image  generate_atten_cylinder.par
echo "===  make attenuation image for SPECT: to make up for the fliprl of SPECTUB"
generate_image generate_atten_cylinder_SPECT.par
fi

# Function to comment out the specific line in a given file and write to a new file
comment_out_line() {
  input_file="$1"
  output_file="$2"
  
  sed '2s/^/;/' "$input_file" > "$output_file"
}

# Paths to the input and output files for uniform cylinder
uniform_input_file="my_uniform_cylinder.hv"
uniform_output_file="my_uniform_cylinder_SPECT.hv"

# Comment out the "modality" line in the uniform image file
comment_out_line "$uniform_input_file" "$uniform_output_file"

# Paths to the input and output files for attenuation image
atten_input_file="my_atten_image_SPECT.hv"
atten_output_file="my_atten_image_SPECT_modified.hv"   

# Comment out the specific line in the attenuation image file
comment_out_line "$atten_input_file" "$atten_output_file"


if [ "$SPECT" -eq 0 ]; then
if [ "$TOF" -eq 0 ]; then
  : ${view_mash:=1}
  : ${span:=2}
  : ${max_rd:=3}
  echo "===  create template sinogram (DSTE with view_mash=${view_mash}, max_ring_diff=${max_rd})"
  template_sino=my_DSTE_3D_vm${view_mash}_span${span}_rd${max_rd}_template.hs
  cat > my_input.txt <<EOF
Discovery STE

$view_mash
n

$span
$max_rd
EOF
else
  : ${view_mash:=2}
  : ${span:=2}
  : ${max_rd:=3}
  : ${tof_mash:=11}
  echo "===  create template sinogram (D690 with view_mash=${view_mash}, TOF_mash=${tof_mash}, max_ring_diff ${max_rd})"
  template_sino=my_D690_3D_vm${view_mash}_span${span}_rd${max_rd}_TOFmash${tof_mash}_template.hs
  cat > my_input.txt <<EOF
Discovery 690

$view_mash
$tof_mash
N

$span
$max_rd
EOF
fi

create_projdata_template  ${template_sino} < my_input.txt > my_create_${template_sino}.log 2>&1
if [ $? -ne 0 ]; then 
  echo "ERROR running create_projdata_template. Check my_create_${template_sino}.log"; exit 1; 
fi

  # fix-up header by insert energy info just before the end
  # trick for awk comes from the www.theunixschool.com
  awk '/END OF INTERFILE/ { print "number of energy windows := 1\nenergy window lower level[1] := 350\nenergy window upper level[1] :=  650\nEnergy resolution := 0.22\nReference energy (in keV) := 511" }1 ' \
      "${template_sino}" > tmp_header.hs
  mv tmp_header.hs "${template_sino}"
  if [ $force_zero_view_offset -eq 1 ]; then
    if [ "$TOF" -eq 1 ]; then
        echo "$0 would need work to be used with both TOF and zero-offset. Exiting"
        exit 1
    fi
    new_template_sino="my_DSTE_3D_rd2_template${suffix}.hs"
    force_view_offset_to_zero.sh "${new_template_sino}" "${template_sino}"
    template_sino="${new_template_sino}"
  fi
fi


# create sinograms
: ${background_value:=10}
if [ "$SPECT" -eq 1 ]; then
  # create SPECT sinograms
  ./simulate_data.sh "my_uniform_cylinder_SPECT.hv" "my_atten_image_SPECT_modified.hv" "SPECT_test_Interfile_header.hs" "${background_value}" "${suffix}"
  if [ $? -ne 0 ]; then
    echo "Error running SPECT simulation"
       exit 1
     fi
else
  ./simulate_data.sh "my_uniform_cylinder.hv" "my_atten_image.hv" "${template_sino}" "${background_value}" "${suffix}"
  if [ $? -ne 0 ]; then
    echo "Error running PET simulation"
    exit 1
  fi
fi
