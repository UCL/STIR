#! /bin/sh -e
# unlists listmode data into span2 and creates randoms estimate from the singles in the listmode data
# Author: Kris Thielemans

# right now randoms stuff only handles single time frame
# we would need a simple loop to cover that case as well.

# directory with some standard .par files
: ${pardir:=~/devel/STIR/examples/GE-Signa-PETMR}

# should get these parameters from command line
: ${INPUT:=LIST0000.BLF}
: ${FRAMES:=frames.fdef}

export INPUT FRAMES

if [ ! -f $FRAMES ]; then
    # make a frame definition file with 1 frame for all the data
    create_fdef_from_listmode.sh frames.fdef $listmode
fi

# create prompt sinograms
OUTPUT=sinospan2 TEMPLATE=${pardir}/template.hs lm_to_projdata ${pardir}/lm_to_projdata.par 

# estimate randoms from singles
construct_randoms_from_GEsingles randomsspan2 "${INPUT}" sinospan2_f1g1d0b0.hs

