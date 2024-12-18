#!/usr/bin/env bash
# (note: need bash for "type -p" below)
# A script to check to see if scatter simulation and estimation give the expected result.
#
#  Copyright (C) 2011, Kris Thielemans
#  Copyright (C) 2013, 2020 University College London
#  This file is part of STIR.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  See STIR/LICENSE.txt for details
#      
# Author Kris Thielemans
#

echo This script should work with STIR version 5.1 and 5.2. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.
echo

if [ $# -eq 1 ]; then
  echo "Prepending $1 to your PATH for the duration of this script."
  PATH=$1:$PATH
fi

command -v generate_image >/dev/null 2>&1 || { echo "generate_image not found or not executable. Aborting." >&2; exit 1; }
echo "Using `command -v generate_image`"
echo "Using `command -v estimate_scatter`"

# first need to set this to the C locale, as this is what the STIR utilities use
# otherwise, awk might interpret floating point numbers incorrectly
LC_ALL=C
export LC_ALL

# find location of scatter parameter files
examples_dir=`stir_config --examples-dir`
scatter_pardir="$examples_dir/samples/scatter_estimation_par_files"
echo "Using scatter parameter files from $scatter_pardir"

./simulate_data_for_tests.sh
if [ $? -ne 0 ]; then
  echo "Error running simulation"
  exit 1
fi

error_log_files=""

echo "===  run scatter simulation"
./simulate_scatter.sh my_scatter_cylinder my_uniform_cylinder.hv my_atten_image.hv scatter_cylinder.hs > my_simulate_scatter.log
if [ $? -ne 0 ]; then
  echo "Error running scatter simulation"
  error_log_files="${error_log_files} my_simulate_scatter.log my_scatter_cylinder*.log"
  echo "Check ${error_log_files}"
  tail ${error_log_files}
  exit 1
fi

echo "===  compare result"
compare_projdata -t .0014 my_scatter_cylinder.hs scatter_cylinder.hs > my_scatter_compare_projdata.log 2>&1
if [ $? -ne 0 ]; then
  echo "Error comparing scatter output."
  error_log_files="${error_log_files} my_simulate_scatter.log my_scatter_cylinder*.log my_scatter_compare_projdata.log"
fi

##### now test scater estimation
# we will first upsample the low-res scatter simulation (with a small
# scale factor to accomodate for multiple scatter) and then generate
# prompt data with scatter.
# Then we run estimate scatter on that data and compare the scatter estimate
# with the original.
echo "===  run scatter upsampling"

cat << EOF > my_norm.par
Bin Normalisation parameters:=
type:= from projdata
  Bin Normalisation From ProjData :=
    normalisation projdata filename:= my_norm.hs
  End Bin Normalisation From ProjData:=
End:=
EOF
# upsample using global factor 1.2 (data-to-fit and weights will actually be ignored)
# tail-fit factors should therefore about around 1.2
upsample_and_fit_single_scatter\
    --min-scale-factor 1.2 \
    --max-scale-factor 1.2 \
    --output-filename my_upsampled_scatter_cylinder.hs \
    --data-to-fit my_prompts.hs \
    --data-to-scale scatter_cylinder.hs \
    --norm my_norm.par \
    --weights my_prompts.hs > my_upsample_and_fit_single_scatter.log 2>&1
if [ $? -ne 0 ]; then
  echo "Error upsampling scatter output."
  error_log_files="${error_log_files} my_upsample_and_fit_single_scatter.log"
  echo "There were errors. Check ${error_log_files}"
  tail -n 80 ${error_log_files}
  exit 1
fi

stir_math -s my_prompts_with_scatter.hs my_prompts.hs my_upsampled_scatter_cylinder.hs

echo "===  run scatter estimation"

## set location of files
## input files
sino_input=my_prompts_with_scatter.hs
atnimg=my_atten_image.hv
NORM=my_norm.hs
acf3d=my_acfs.hs
randoms3d=my_randoms.hs
## recon settings during scatter estimation
# adjust for your scanner (needs to divide number of views/4 as usual)
scatter_recon_num_subsets=7
scatter_recon_num_subiterations=7
# scatter settings
num_scat_iters=3
## filenames for output
mask_projdata_filename=my_mask.hs
mask_image=my_mask_image.hv
scatter_prefix=my_estimated_scatter
total_additive_prefix=my_addsino

export scatter_pardir
export sino_input atnimg NORM acf3d randoms3d
export num_scat_iters scatter_recon_num_subsets scatter_recon_num_subiterations
export mask_projdata_filename mask_image scatter_prefix total_additive_prefix

estimate_scatter $scatter_pardir/scatter_estimation.par > my_estimate_scatter.log 2>&1
if [ $? -ne 0 ]; then
  echo "Error estimating scatter."
  error_log_files="${error_log_files} my_estimate_scatter.log"
  echo "There were errors. Check ${error_log_files}"
  tail -n 80 ${error_log_files}
  exit 1
fi


echo "===  compare result (up to 7%)"
# threshold needs to be a bit high as scatter_cylinder.hs was generated without random sampling
compare_projdata -t .07 my_estimated_scatter_3.hs my_upsampled_scatter_cylinder.hs > my_estimate_scatter_compare_projdata.log 2>&1
if [ $? -ne 0 ]; then
  echo "Error comparing scatter output."
  error_log_files="${error_log_files} my_estimate_scatter*.log"
fi

if [ -z "${error_log_files}" ]; then
 echo "All tests OK!"
 echo "You can remove all output using \"rm -f my_*\""
 exit 0
else
 echo "There were errors. Check ${error_log_files}"
 tail -n 80 ${error_log_files}
 exit 1
fi

