#! /bin/sh
# A script to do a simplistic analytic simulation, as used by test_simulate_and_recon.sh
#
#  Copyright (C) 2011 - 2011-01-14, Hammersmith Imanet Ltd
#  Copyright (C) 2011-07-01 - 2011, Kris Thielemans
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
set -e

if [ $# -ne 3 ]; then
  echo "Usage: `basename $0` emission_image attenuation_image template_sino"
  echo "Creates my_prompts.hs my_randoms.hs my_acfs.hs my_additive_sinogram.hs "
  exit 1
fi

emission_image=$1
atten_image=$2
template_sino=$3

echo "===  create ACFs"
calculate_attenuation_coefficients --ACF my_acfs.hs ${atten_image} ${template_sino} > my_create_acfs.log 2>&1
if [ $? -ne 0 ]; then 
  echo "ERROR running calculate_attenuation_coefficients. Check my_create_acfs.log"; exit 1; 
fi

echo "===  create line integrals"
forward_project my_line_integrals.hs  ${emission_image} ${template_sino} forward_projector_proj_matrix_ray_tracing.par > my_create_line_integrals.log 2>&1
if [ $? -ne 0 ]; then 
  echo "ERROR running forward_project. Check my_create_line_integrals.log"; exit 1; 
fi


echo "=== create constant randoms background"
${INSTALL_DIR}stir_math -s --including-first \
         --times-scalar 0 --add-scalar 10 \
         my_randoms my_line_integrals.hs
if [ $? -ne 0 ]; then 
  echo "ERROR running stir_math"; exit 1; 
fi

echo "===  create prompts"
correct_projdata uncorrect_projdata_simulation.par > my_create_prompts.log 2>&1
if [ $? -ne 0 ]; then 
  echo "ERROR running correct_projdata. Check my_create_prompts.log"; exit 1; 
fi

# could call poisson_noise here

echo "===  create additive sinogram for reconstruction"
# need randoms (and scatter) multiplied by ACF and norm (but we don't have a norm here)
stir_math -s --mult  my_additive_sinogram.hs my_randoms.hs my_acfs.hs
if [ $? -ne 0 ]; then 
  echo "ERROR running stir_math"; exit 1; 
fi

echo "Done creating simulated data"
