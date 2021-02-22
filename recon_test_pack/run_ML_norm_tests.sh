#! /bin/sh
# A script to check to see if ML norm estimation gives the expected result.
#
# This tests only at the most basic level currently:
# - create data which is a multiple of a "model"
# - find norm factors from the data (efficiencies should be sqrt(scale-factor)
# - apply to model
# - check if it's equal to the data

#  Copyright (C) 2013, 2020, 2021 University College London
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

echo This script should work with STIR version 5.0. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.
echo

if [ $# -eq 1 ]; then
  echo "Prepending $1 to your PATH for the duration of this script."
  PATH=$1:$PATH
fi

command -v find_ML_normfactors3D >/dev/null 2>&1 || { echo "find_ML_norm_factors3D not found or not executable. Aborting." >&2; exit 1; }
echo "Using `command -v find_ML_normfactors3D`"
echo "Using `command -v apply_normfactors3D`"
echo "Using `command -v stir_math`"

# We will use an existing file as a test.
# Of course, it doesn't make sense to run normalisation using scatter data,
# but this is just a self-consistency check at present.
input=scatter_cylinder.hs
if [ -r $input ]; then
    :
else
    echo "Error: $input does not exist" 1>&2
fi
# first need to set this to the C locale, as this is what the STIR utilities use
# otherwise, awk might interpret floating point numbers incorrectly
LC_ALL=C
export LC_ALL

error_log_files=""

model_data=$input
measured_data=my_norm_test.hs
stir_math -s --including-first --times-scalar 4  $measured_data $input >/dev/null 2>&1

num_iters=2
num_eff_iters=3
check_num_iters=$num_iters
check_num_eff_iters=$num_eff_iters
echo "===  run ML norm estimation"
prog=find_ML_normfactors3D
$prog  my_norm $measured_data $model_data $num_iters $num_eff_iters > my_${prog}.log 2>&1
if [ $? -ne 0 ]; then
  echo "Error running $prog"
  error_log_files="${error_log_files} my_${prog}*.log"
  echo "Check ${error_log_files}"
  tail ${error_log_files}
  exit 1
fi

echo "===  apply ML"
prog=apply_normfactors3D
$prog my_norm_check my_norm $model_data 1 $check_num_iters $check_num_eff_iters > my_${prog}.log 2>&1
if [ $? -ne 0 ]; then
  echo "Error running $prog"
  error_log_files="${error_log_files} my_${prog}*.log"
  echo "Check ${error_log_files}"
  tail ${error_log_files}
  exit 1
fi

echo "===  compare result"
compare_projdata  my_norm_check.hs $measured_data > my_norm_check_compare_projdata.log 2>&1
if [ $? -ne 0 ]; then
  echo "Error comparing output."
  error_log_files="${error_log_files}"
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

