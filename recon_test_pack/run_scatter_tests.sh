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
# Authors Kris Thielemans
#        Nikos Efthimiou
# 

# Scripts should exit with error code when a test fails:
if [ -n "$TRAVIS" ]; then
    # The code runs inside Travis
    set -e
fi

echo This script should work with STIR version ">"3.0. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.
echo

if [ $# -eq 1 ]; then
  echo "Prepending $1 to your PATH for the duration of this script."
  PATH=$1:$PATH
fi

# first delete any files remaining from a previous run
rm -f my_* *.log

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

echo "===  Downsample the attenuation image"
# This will be used to "select" scatter points in the simulation.
# Note: the more downsampling, the faster, but less accurate of course.
# Downsampling factors and final size are currently hard-wired in this script.
# You'd have to adjust these for your data.
ATTEN_IMAGE=my_zoomed_my_atten_image.hv
zoom_z=.16666667
zoom_xy=0.25
new_voxels_z=8
new_voxels_xy=33
zoom_image ${ATTEN_IMAGE} my_atten_image.hv ${new_voxels_xy} ${zoom_xy} 0 0 ${new_voxels_z} ${zoom_z} 0
if [ $? -ne 0 ]; then
  echo "Error running zoom_image"
  exit 1
fi
# scale image back to appropriate units (cm^-1)
stir_math --accumulate  --times-scalar ${zoom_xy}  --times-scalar ${zoom_xy} --times-scalar ${zoom_z}  --including-first ${ATTEN_IMAGE}
if [ $? -ne 0 ]; then
  echo "Error running stir_math"
  exit 1
fi

export ATTEN_IMAGE
echo "===  run scatter simulation (new)"
simulate_scatter scatter_simulation_new.par > my_simulate_scatter_new.log
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

