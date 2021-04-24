#! /bin/sh
# A script to do a simplistic analytic simulation, as used by test_simulate_and_recon.sh
#
#  Copyright (C) 2011 - 2011-01-14, Hammersmith Imanet Ltd
#  Copyright (C) 2011-07-01 - 2011, Kris Thielemans
#  This file is part of STIR.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  See STIR/LICENSE.txt for details
#      
# Author Kris Thielemans
# 

if [ $# -lt 3  -o $# -gt 5 ]; then
  echo "Usage: `basename $0` emission_image attenuation_image template_sino  [ background_value [suffix] ]"
  echo "Creates my_prompts$suffix.hs my_randoms$suffix.hs my_acfs$suffix.hs my_additive_sinogram$suffix.hs"
  exit 1
fi

emission_image=$1
atten_image=$2
template_sino=$3
if [ $# -gt 3 ]; then
  background_value=$4
else
  background_value=10
fi
if [ $# -gt 4 ]; then
  suffix=$5
fi
echo "===  create ACFs"
calculate_attenuation_coefficients --ACF my_acfs$suffix.hs ${atten_image} ${template_sino} > my_create_acfs.log 2>&1
if [ $? -ne 0 ]; then 
  echo "ERROR running calculate_attenuation_coefficients. Check my_create_acfs.log"; exit 1; 
fi

echo "===  create line integrals"
forward_project my_line_integrals$suffix.hs  ${emission_image} ${template_sino} forward_projector_proj_matrix_ray_tracing.par > my_create_line_integrals.log 2>&1
if [ $? -ne 0 ]; then 
  echo "ERROR running forward_project. Check my_create_line_integrals.log"; exit 1; 
fi


echo "=== create constant randoms background"
${INSTALL_DIR}stir_math -s --including-first \
         --times-scalar 0 --add-scalar $background_value \
         my_randoms$suffix my_line_integrals$suffix.hs
if [ $? -ne 0 ]; then 
  echo "ERROR running stir_math"; exit 1; 
fi

echo "===  create norm factors"
# currently just 1 as not used in rest of script yet.
stir_math -s --including-first \
          --times-scalar 0 --add-scalar 1 my_norm$suffix.hs my_line_integrals$suffix.hs

echo "===  create prompts"
export suffix # used in the .par file to determine filenames
correct_projdata uncorrect_projdata_simulation.par > my_create_prompts.log 2>&1
if [ $? -ne 0 ]; then 
  echo "ERROR running correct_projdata. Check my_create_prompts.log"; exit 1; 
fi

# could call poisson_noise here

echo "===  create additive sinogram for reconstruction"
# need randoms (and scatter) multiplied by ACF and norm (but we don't have a norm here)
stir_math -s --mult  my_additive_sinogram$suffix.hs my_randoms$suffix.hs my_acfs$suffix.hs
if [ $? -ne 0 ]; then 
  echo "ERROR running stir_math"; exit 1; 
fi

echo "Done creating simulated data"
