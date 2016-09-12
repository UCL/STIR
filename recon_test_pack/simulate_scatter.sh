#! /bin/sh
# A script to do a simplistic analytic simulation of scatter as used by the recon_test_pack
#
#  Copyright (C) 2013 University College London
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
#         Nikos Efthimiou
# 

# Scripts should exit with error code when a test fails:
if [ -n "$TRAVIS" ]; then
    # The code runs inside Travis
    set -e
fi

if [ $# -ne 4 ]; then
  echo "Usage: `basename $0` output_prefix emission_image attenuation_image template_sino"
  exit 1
fi

OUTPUT_PREFIX=$1
ACTIVITY_IMAGE=$2
org_atten_image=$3
TEMPLATE=$4
ATTEN_IMAGE=my_zoomed_${org_atten_image}

zoom_z=.16666667
zoom_xy=0.25
new_voxels_z=8
new_voxels_xy=33
zoom_image ${ATTEN_IMAGE} ${org_atten_image} ${new_voxels_xy} ${zoom_xy} 0 0 ${new_voxels_z} ${zoom_z} 0
if [ $? -ne 0 ]; then
  echo "Error running zoom_image"
  exit 1
fi
stir_math --accumulate  --times-scalar ${zoom_xy}  --times-scalar ${zoom_xy} --times-scalar ${zoom_z}  --including-first ${ATTEN_IMAGE}
if [ $? -ne 0 ]; then
  echo "Error running stir_math"
  exit 1
fi

export OUTPUT_PREFIX ACTIVITY_IMAGE ATTEN_IMAGE TEMPLATE

simulate_scatter scatter_simulation.par 2> ${OUTPUT_PREFIX}_stderr.log
if [ $? -ne 0 ]; then
  echo "Error running simulate_scatter"
  exit 1
fi

echo "Done creating simulated data"
