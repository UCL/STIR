#! /bin/sh
# A script to do a simplistic analytic simulation, as used by test_simulate_and_recon.sh
#
#  Copyright (C) 2011 - 2011-01-14, Hammersmith Imanet Ltd
#  Copyright (C) 2011-07-01 - 2011, Kris Thielemans
#  Copyright (C) 2024 University College London
#  This file is part of STIR.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  See STIR/LICENSE.txt for details
#      
# Author Kris Thielemans
# Author Dimitra Kyriakopoulou

# Set single-threaded execution
#export OMP_NUM_THREADS=1

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


# Check if the suffix is "_SPECT"
if [ "$suffix" = "_SPECT" ]; then

	echo "=== create line integrals for SPECT sinogram"
	forward_project "my_sino${suffix}.hs" "${emission_image}" "${template_sino}" forward_project_SPECTUB.par > "my_create_sino${suffix}.log" 2>&1
	if [ $? -ne 0 ]; then 
			echo "ERROR running forward_project for SPECT sinogram. Check my_create_sino${suffix}.log"; exit 1;
	fi


	echo "===  create line integrals for SRT2DSPECT attenuation projection"
	forward_project "my_attenuation_sino$suffix.hs" "${atten_image}" "${template_sino}" > "my_create_attenuation_sino${suffix}.log" 2>&1
	if [ $? -ne 0 ]; then 
		echo "ERROR running forward_project for attenuation sinogram. Check my_create_attenuation_sino${suffix}.log"; exit 1;
  fi

else

	 echo "===  create ACFs"
	calculate_attenuation_coefficients --ACF my_acfs$suffix.hs ${atten_image} ${template_sino} forward_projector_proj_matrix_ray_tracing.par > my_create_acfs${suffix}.log 2>&1
	if [ $? -ne 0 ]; then 
		echo "ERROR running calculate_attenuation_coefficients. Check my_create_acfs${suffix}.log"; exit 1;
	fi

	echo "===  create line integrals"
	forward_project my_line_integrals$suffix.hs  ${emission_image} ${template_sino} forward_projector_proj_matrix_ray_tracing.par > my_create_line_integrals${suffix}.log 2>&1
	if [ $? -ne 0 ]; then 
		echo "ERROR running forward_project. Check my_create_line_integrals${suffix}.log"; exit 1;
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
		        --times-scalar 0 --add-scalar 1 my_norm$suffix.hs my_acfs$suffix.hs

	echo "===  create prompts"
	INPUT=my_line_integrals${suffix}.hs OUTPUT=my_prompts${suffix}.hs \
		   MULT=my_acfs${suffix}.hs \
		   RANDOMS=my_randoms${suffix}.hs \
		   correct_projdata uncorrect_projdata.par > my_create_prompts${suffix}.log 2>&1
	if [ $? -ne 0 ]; then 
		echo "ERROR running correct_projdata. Check my_create_prompts${suffix}.log"; exit 1;
	fi

	# could call poisson_noise here

	echo "===  create additive sinogram for reconstruction"
	# need randoms (and scatter) multiplied by ACF and norm (but we don't have a norm here yet)
	# need to use correct_projdata as in TOF, ACF/norm is non-TOF, so stir_math will fail
	INPUT=my_randoms${suffix}.hs OUTPUT=my_additive_sinogram${suffix}.hs \
	MULT=my_acfs${suffix}.hs \
		   correct_projdata correct_projdata_norm_only.par > my_create_additive_sino${suffix}.log 2>&1
	if [ $? -ne 0 ]; then 
		echo "ERROR running correct_projdata. Check my_create_additive_sino${suffix}.log"; exit 1;
	fi
fi

echo "Done creating simulated data"
