#! /bin/sh -e
# Example script to reconstruct GE-Signa PET/MR data with randoms, norm and scatter.
# Currently supposes you have randoms estimated already.
# Default filenames are for output of unlist_and_randoms.sh, and for the NEMA demo files
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

: ${num_subsets:=14}
: ${num_subiters:=42}

export sino_input randoms3d # used by scatter_estimation.par

# RDF9 norm file 
: ${RDFNORM:=norm3d}

# we will put most intermediate files in a separate directory
# note that the scatter code will also create an `extras` folder in the current directory
mkdir -p output

# output name (or input if it exists already) for normalisation sinogram
: ${norm_sino_prefix:=output/fullnormfactorsspan2}
${pardir}/create_norm_projdata.sh "${norm_sino_prefix}" "${RDFNORM}" "${sino_input}"

# use some more memorable names
data3d=${sino_input}
norm3d=${norm_sino_prefix}.hs

echo "Creating additive sinogram for the reconstruction (randoms only here)"
additive_sino=output/add_sino.hs
stir_math -s ${additive_sino}  "${randoms3d}" "${norm3d}"

### do an image reconstruction

# FBP recon (currently commented out, not checked)
#stir_math -s --mult precorrected_data3d.hs ${data3d} ${norm3d}
#stir_subtract -s --accumulate  precorrected_data3d.hs ${additive_sino}
#ZOOM=.4 INPUT=precorrected_data3d.hs OUTPUT=final_activity_image_3d FBP3DRP FBP3DRP.par

echo "Running OSMAPOSL"
echo "Log will be in NAC_image.log"
INPUT="${data3d}" OUTPUT=NAC_image NORM="${norm3d}" ADDSINO="${additive_sino}" \
    SUBSETS=$num_subsets SUBITERS=$num_subiters SAVEITERS=$num_subsets SENS=output/subset_sens_NAC RECOMP_SENS=1 \
     OSMAPOSL ${pardir}/OSMAPOSLbackground.par 2>&1 | tee NAC_image.log
