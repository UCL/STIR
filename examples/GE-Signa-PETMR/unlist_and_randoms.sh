#! /bin/sh -e
# unlists listmode data into span2 and creates MLrandoms estimate from the listmode data
# Author: Kris Thielemans

# right now randoms stuff only handles single time frame

# directory with some standard .par files
: ${pardir:=~/SIRF/workspace}

# should get these parameters from command line
: ${INPUT:=LIST0000.BLF}
: ${FRAMES:=frames.fdef}

export INPUT FRAMES

# create prompt sinograms
OUTPUT=sinospan2 TEMPLATE=${pardir}/template.hs lm_to_projdata ${pardir}/lm_to_projdata.par 

# estimate randoms from singles
construct_randoms_from_GEsingles randomsspan2 ${INPUT} template.hs 