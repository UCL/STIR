#! /bin/sh
# A script to simulate some data used by other tests.
# Careful: run_scatter_tests.sh compares with previously generated data
# so you cannot simply modify this one with adjusting that as well.
#
# This script is not intended to be run on its own!
#
#  Copyright (C) 2011 - 2011-01-14, Hammersmith Imanet Ltd
#  Copyright (C) 2011-07-01 - 2011, Kris Thielemans
#  Copyright (C) 2014,2020 University College London
#  This file is part of STIR.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  See STIR/LICENSE.txt for details
#      
# Author Kris Thielemans
# 

echo This script should work with STIR version 5.x. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.
echo

command -v generate_image >/dev/null 2>&1 || { echo "generate_image not found or not executable. Aborting." >&2; exit 1; }
echo "Using `command -v generate_image`"

force_zero_view_offset=0
suffix=""
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
  elif test "$1" = "--suffix"
  then
    suffix="$2"
    shift 1
  elif test "$1" = "--help"
  then
    echo "Usage: `basename $0` [--force_zero_view_offset]  [--suffix sometext] [install_dir]"
    echo "(where [] means that an argument is optional)"
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


echo "===  make emission image"
generate_image  generate_uniform_cylinder.par
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

# fix-up header by insert energy info just before the end
# trick for awk comes from the www.theunixschool.com
awk '/END OF INTERFILE/ { print "number of energy windows := 1\nenergy window lower level[1] := 350\nenergy window upper level[1] :=  650\nEnergy resolution := 0.22\nReference energy (in keV) := 511" }1 ' \
    ${template_sino} > tmp_header.hs
mv tmp_header.hs ${template_sino}

if [ $force_zero_view_offset -eq 1 ]; then
  new_template_sino=my_DSTE_3D_rd2_template$suffix.hs
  force_view_offset_to_zero.sh ${new_template_sino} ${template_sino}
  template_sino=${new_template_sino}
fi

# create sinograms
./simulate_data.sh my_uniform_cylinder.hv my_atten_image.hv ${template_sino} 10 ${suffix}
if [ $? -ne 0 ]; then
  echo "Error running simulation"
  exit 1
fi
