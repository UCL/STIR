#! /bin/sh
# A script to check to see if Time Of Flight data are binned and used properly
#
#  Copyright (C) 2016, University of Leeds
#  Copyright (C) 2017, University of Hull
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
# Author Nikos Efthimiou
# 

# Scripts should exit with error code when a test fails:
if [ -n "$TRAVIS" ]; then
    # The code runs inside Travis
    set -e
fi

echo This script should work with STIR version '>'3.0. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.
echo
 
if [ $# -eq 1 ]; then
  echo "Prepending $1 to your PATH for the duration of this script."
  PATH=$1:$PATH
fi

# first need to set this to the C locale, as this is what the STIR utilities use
# otherwise, awk might interpret floating point numbers incorrectly
LC_ALL=C
export LC_ALL

echo "===  create template sinogram. We'll use a test_scanner which is small and 
has TOF info"
template_sino=my_test_scanner_template.hs
cat > my_input.txt <<EOF
test_scanner
1
82
N



EOF
create_projdata_template  ${template_sino} < my_input.txt > my_create_${template_sino}.log 2>&1
if [ $? -ne 0 ]; then 
  echo "ERROR running create_projdata_template. Check my_create_${template_sino}.log"; exit 1; 
fi

export INPUT_ROOT_FILE=test_PET_GATE.root
export EXCLUDE_RANDOM=1
export EXCLUDE_SCATTERED=1

INPUT=root_header.hroot TEMPLATE=$template_sino OUT_PROJDATA_FILE=my_tof_sinogram lm_to_projdata  --test_timing_positions lm_to_projdata.par > my_write_TOF_values_${template_sino}.log 2>&1

if [ $? -ne 0 ]; then 
  echo "ERROR running lm_to_projdata  --test_timing_positions. Check my_write_TOF_values_${template_sino}.log"; exit 1; 
fi

echo "Comparing values in TOF sinogram ..."
list_projdata_info --all my_tof_sinogram_f179g1d0b0.hs > my_sino_values_$template_sino.log 2>&1
if [ $? -ne 0 ]; then 
  echo "ERROR running list_projdata_info. Check my_sino_values_$template_sino.log"; 
  exit 1; 
fi


TOF_bins=$(grep 'Total number of timing positions' my_sino_values_$template_sino.log | awk -F ':' '{ print  $2 }')
echo "Total number of TOF bins:" $TOF_bins

Timming_Locations=$(grep 'Timing location' my_sino_values_$template_sino.log | awk -F ':' '{ print  $2 }')
echo "Timming_Locations:" $Timming_Locations

Data_mins=$(grep 'Data min' my_sino_values_$template_sino.log | awk -F ':' '{ print  $2 }')
echo "Data mins:" $Data_mins

Data_maxs=$(grep 'Data max' my_sino_values_$template_sino.log | awk -F ':' '{ print  $2 }')
echo "Data maxs:" $Data_maxs

for i in $(seq 5)
do
  if [ $(( $(($(($i-1)) - $((TOF_bins/2)))) - $((Data_mins[$i])))) -ne 0 ]; then
    echo "Wrong values in TOF sinogram. Error. $(( $(($(($i-1)) - $((TOF_bins/2)))) - $((Data_mins[$i]))))"
    exit 1
    fi
done


echo
echo '--------------- End of Time-Of-Flight tests -------------'
echo
echo "Everything seems to be fine !"
echo 'You could remove all generated files using "rm -f my_* *.log"'
exit 0

