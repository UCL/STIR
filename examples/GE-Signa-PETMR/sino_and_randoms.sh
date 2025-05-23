#! /bin/sh -e
# copies prompts data from a GE sinogram RDF and creates randoms estimate from the singles in the RDF
# output names are the same as the unlist_and_randoms.sh script for convenience
# Author: Kris Thielemans
# SPDX-License-Identifier: Apache-2.0
# See STIR/LICENSE.txt for details

# should get these parameters from command line
: ${INPUT:=rdf_f1b1.rdf}

# create prompt sinograms
# copy prompts to Interfile to get round a limitation of the current
# RDF reader (it doesn't have get_sinogram(), and is rather slow)
stir_math -s sinospan2_f1g1d0b0.hs "${INPUT}"

# estimate randoms from singles
construct_randoms_from_GEsingles randomsspan2_f1g1d0b0 "${INPUT}" sinospan2_f1g1d0b0.hs

