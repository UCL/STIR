#! /bin/sh
# An example script to do a simplistic analytic simulation
# It will call simulate_scatter.sh and various STIR utilities, so all this needs to be in your path
#
#  Copyright (C) 2011 - 2011-01-14, Hammersmith Imanet Ltd
#  Copyright (C) 2011-07-01 - 2011-08-23, Kris Thielemans
#  Copyright (C) 2013-2014 University College London
#  This file is part of STIR.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  See STIR/LICENSE.txt for details
#      
# Author Kris Thielemans
# 

if [ $# -ne 5 ]; then
  echo "Usage: `basename $0` emission_image attenuation_image template_sino scatter.par scatter_template.hs"
  echo "Creates my_prompts.hs  my_multfactors.hs  my_additive_sinogram.hs "
  echo "and some other intermediate files"
  exit 1
fi

emission_image=$1
atten_image=$2
template_sino=$3
scatter_params="$4 $5"

echo "===  create line integrals"
# Call forward_project to compute line integrals
# If you have a STIR version prior to 3.0, you can uncomment the following line (and comment the one 
# after that) to use fwdtest
#fwdtest my_line_integrals.hs ${template_sino} ${emission_image} ../forward_projector_proj_matrix_ray_tracing.par </dev/null > my_create_line_integrals.log 2>&1
forward_project my_line_integrals.hs ${emission_image}  ${template_sino} ../forward_projector_proj_matrix_ray_tracing.par > my_create_line_integrals.log 2>&1
if [ $? -ne 0 ]; then 
  echo "ERROR running forward_project. Check my_create_line_integrals.log"; exit 1; 
fi

echo "===  create ACFs"
calculate_attenuation_coefficients --ACF my_acfs.hs ${atten_image} ${template_sino} > my_create_acfs.log 2>&1
if [ $? -ne 0 ]; then 
  echo "ERROR running calculate_attenuation_coefficients. Check my_create_acfs.log"; exit 1; 
fi

echo "===  create normalisation factors"
# For illustration purposes, we will use include a normalisation file but it is currently
# just set to a constant sinogram 
# (all bins set to 3.4 here, just to check if it's handled properly in the reconstruction script)
stir_math -s --including-first \
         --times-scalar 0 --add-scalar 3.4 \
         my_norm.hs my_acfs.hs
if [ $? -ne 0 ]; then 
  echo "ERROR running stir_math for norm"; exit 1; 
fi

echo "===  multiply ACF and norm"
stir_math -s --mult my_multfactors.hs my_norm.hs my_acfs.hs
if [ $? -ne 0 ]; then 
  echo "ERROR running stir_math for multiplying norm and ACFs"; exit 1; 
fi



echo "===  create constant randoms background"
# For illustration purposes, we will have a constant randoms background.
# The number 10 is used, but this is arbitrary (and it's going to be modified below anyway).
# TODO compute sensible randoms-to-true background here
stir_math -s --including-first \
         --times-scalar 0 --add-scalar 10 \
         my_randoms.hs my_line_integrals.hs
# We divide by the norm here to put some efficiency pattern on the data.
# This isn't entirely accurate as a "proper" norm contains geometric effects
# which shouldn't be in the randoms. Moreover, the norm can contain a global scaling
# factor (e.g. for calibration) so this division will actually scale the randoms as well.
# In particular, in the current script, it will divide the randoms by 3.4!
# Ideally we'd fix this but this is supposed to be a "simple" simulation :-;
stir_divide -s --accumulate \
         my_randoms.hs my_norm.hs
if [ $? -ne 0 ]; then 
  echo "ERROR running stir_math"; exit 1; 
fi
# We've finished constructing my_randoms.hs


echo "===  create scatter"
# we could just use zero scatter
#stir_math -s --including-first \
#         --times-scalar 0 --add-scalar 0 \
#         my_scatter.hs my_line_integrals.hs

simulate_scatter.sh my_scatter ${emission_image} ${atten_image} my_norm.hs ${scatter_params}
if [ $? -ne 0 ]; then 
  echo "ERROR running simulate_scatter"; exit 1; 
fi

echo "===  create additive sinogram for reconstruction"
# model is
#   data = 1/(ACF*norm) * ( fwd(image) + additive_sinogram)
#
# Therefore, we need (randoms+scatter)  multiplied by ACF and norm 
stir_math -s   my_additive_sinogram.hs  my_randoms.hs my_scatter.hs
stir_math -s --mult --accumulate  my_additive_sinogram.hs my_multfactors.hs
if [ $? -ne 0 ]; then 
  echo "ERROR running stir_math for additive sinogram"; exit 1; 
fi

echo "===  create prompts"
stir_math -s my_prompts.hs my_line_integrals.hs my_additive_sinogram.hs 
stir_divide -s --accumulate  my_prompts.hs my_multfactors.hs

echo "===  create zero sinogram"
# for convenience of the scripts
stir_math -s --including-first \
         --times-scalar 0 \
         my_zeros.hs my_line_integrals.hs

echo "Done creating simulated data"
