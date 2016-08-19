#! /bin/sh
# A script to check to see if scatter simulation gives the expected result.
#
#  Copyright (C) 2011, Kris Thielemans
#  Copyright (C) 2013, University College London
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

echo This script should work with STIR version 2.3, 2.4 and 3.0. If you have
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

error_log_files=""

echo "===  make emission image"
generate_image  generate_uniform_cylinder.par
echo "===  use that as template for attenuation"
stir_math --including-first --times-scalar .096 my_atten_image.hv my_uniform_cylinder.hv

echo "===  run scatter simulation"
./simulate_scatter.sh my_scatter_cylinder my_uniform_cylinder.hv my_atten_image.hv scatter_cylinder.hs > my_simulate_scatter.log
if [ $? -ne 0 ]; then
  echo "Error running scatter simulation"
  error_log_files="${error_log_files} my_simulate_scatter.log my_scatter_cylinder*.log"
  echo "Check ${error_log_files}"
  exit 1
fi

echo "===  compare result"
# we need a fairly large threshold (4%) as scatter points are chosen randomly
compare_projdata -t .04 my_scatter_cylinder.hs scatter_cylinder.hs > my_scatter_compare_projdata.log 2>&1
if [ $? -ne 0 ]; then
  echo "Error comparing scatter output."
  error_log_files="${error_log_files} my_scatter_compare_projdata.log"
fi

if [ -z "${error_log_files}" ]; then
 echo "All tests OK!"
 echo "You can remove all output using \"rm -f my_*\""
else
 echo "There were errors. Check ${error_log_files}"
fi

