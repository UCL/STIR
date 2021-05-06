#! /bin/sh
# a script to simulate scatter for PET
# This will use the single-scatter estimation routine.
# WARNING: it relies on a scatter.par file using the following variables:
#   ${ACTIVITY_IMAGE}, ${ATTEN_IMAGE}, ${TEMPLATE}, ${OUTPUT_PREFIX}
# WARNING: image downsampling factors and final size are currently hard-wired in this script.
#
#  Copyright (C) 2013 University College London
#  This file is part of STIR.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  See STIR/LICENSE.txt for details
#      
# Author Kris Thielemans
# 

if [ $# -ne 6 ]; then
  echo "Usage: `basename $0` output_prefix emission_image attenuation_image normalisation_sino scatterparfile scatter_low_res_template_sino "
  exit 1
fi

output=$1
ACTIVITY_IMAGE=$2
ATTEN_IMAGE=$3
norm=$4
scatterparfile=$5
TEMPLATE=$6

#### compute low resolution scatter
OUTPUT_PREFIX=${output}_low_res
export OUTPUT_PREFIX ACTIVITY_IMAGE ATTEN_IMAGE TEMPLATE
simulate_scatter ${scatterparfile} 2> ${OUTPUT_PREFIX}_stderr.log
if [ $? -ne 0 ]; then
  echo "Error running simulate_scatter"
  exit 1
fi

#### upsample the low-resolution estimate
# We will use a trick here and use the upsample_and_fit_single_scatter executable.
# This is normally designed to do "tail-fitting" of the scatter estimate to the measured data
# Here we force the scale factor to be 1 (as estimate_scatter uses correct units for single scatter)
# That means that "weights" and "data-to-fit" are actually ignored (but we have to specify them anyway)
upsample_and_fit_single_scatter  --min-scale-factor 1 --max-scale-factor 1 \
    --output-filename ${output}_pre_norm.hs --data-to-fit ${norm} --data-to-scale ${OUTPUT_PREFIX}.hs --weights ${norm} 2> ${output}_stderr.log
if [ $? -ne 0 ]; then
  echo "Error upsampling scatter"
  exit 1
fi

#### divide by norm to be consistent with measured data
# STIR scatter simulation produces "normalised" data, so we undo that here
stir_divide -s ${output}.hs ${output}_pre_norm.hs ${norm}

echo "Done creating simulated scatter data"
