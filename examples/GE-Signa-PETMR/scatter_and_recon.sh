#! /bin/sh -e
# Example script to reconstruct GE-Signa PET/MR data with randoms, norm and scatter.
# Currently supposes you have randoms estimated already.
# Default filenames are for output of unlist_and_randoms.sh.
# Author: Kris Thielemans
# Edits: Ander Biguri
# SPDX-License-Identifier: Apache-2.0
# See STIR/LICENSE.txt for details

# directory with some standard .par files
: ${pardir:=$(dirname $0)}
# convert to absolute path (assumes that it exists), from DVK's answer on stackexchange
pardir=`cd "$pardir";pwd`

# names are from unlist_and_randoms
: ${sino_input:=sinospan2_f1g1d0b0.hs}
: ${randoms3d:=randomsspan2_f1g1d0b0.hs}
export sino_input randoms3d # used by scatter_estimation.par

: ${num_subsets:=14}
: ${num_subiters:=42}

# RDF9 norm file 
# note: variable name is used in correct_projdata.par
: ${RDFNORM:=norm3d}
export RDFNORM

: ${atnimg:=att.hv}
export atnimg # used in scatter_estimation.par

# we will put most intermediate files in a separate directory
# note that the scatter code will also create an `extras` folder in the current directory
mkdir -p output

# output name (or input if it exists already) for normalisation sinogram
: ${norm_sino_prefix:=output/fullnormfactorsspan2}
${pardir}/create_norm_projdata.sh ${norm_sino_prefix} ${RDFNORM} ${sino_input}

: ${acf3d:=output/acf.hs} # will be created if it doesn't exist yet
export acf3d  # used in scatter_estimation.par

# image 0 everywhere where activity is zero, 1 elsewhere
: ${mask_image:=output/mask_image.hv}  # will be created by masking the attenuation image

# use some more memorable names
data3d=${sino_input}
norm3d=${norm_sino_prefix}.hs

# compute 3D ACFs
if [ -r ${acf3d} ]; then
  echo "Reusing existing ACFs ${acf3d}"
else
  echo "Creating ACFs ${acf3d}"
  calculate_attenuation_coefficients --PMRT  --ACF ${acf3d} ${atnimg} ${data3d} 
fi

### estimate scatter

echo "Estimating scatter (be patient). Log saved in output/scatter.log"

# filename-prefix for additive sino (i.e. "precorrected" sum of scatter and randoms)
total_additive_prefix=output/total_additive
num_scat_iters=3
scatter_pardir=${pardir}/../samples/scatter_estimation_par_files
# you might have to change this for a different scanner than the Signa PET/MR
# (it needs to be a divisor of the number of views)
scatter_recon_num_subiterations=$num_subsets
scatter_recon_num_subsets=$num_subsets
export scatter_pardir
export num_scat_iters total_additive_prefix
export scatter_recon_num_subiterations scatter_recon_num_subsets
NORM=${norm3d} mask_image=${mask_image} mask_projdata_filename=output/mask2d.hs scatter_prefix=output/scatter \
    estimate_scatter $scatter_pardir/scatter_estimation.par 2>&1 | tee output/scatter.log 
# output for recon
additive_sino=${total_additive_prefix}_${num_scat_iters}.hs

### do an image reconstruction

stir_math -s --mult output/mult_factors_3d.hs  ${acf3d} ${norm3d}

# FBP recon (currently commented out, not checked)
#stir_math -s --mult precorrected_data3d.hs ${data3d} mult_factors_3d.hs
#stir_subtract -s --accumulate  precorrected_data3d.hs ${additive_sino}
#ZOOM=.4 INPUT=precorrected_data3d.hs OUTPUT=final_activity_image_3d FBP3DRP FBP3DRP.par

echo "Running OSMAPOSL"
echo "Log will be in final_activity_image.log"
INPUT=${data3d} OUTPUT=final_activity_image NORM=output/mult_factors_3d.hs ADDSINO=${additive_sino} \
     SUBSETS=$num_subsets SUBITERS=$num_subiters SAVEITERS=$num_subsets SENS=output/subset_sens RECOMP_SENS=1 \
     OSMAPOSL ${pardir}/OSMAPOSLbackground.par 2>&1 | tee final_activity_image.log
