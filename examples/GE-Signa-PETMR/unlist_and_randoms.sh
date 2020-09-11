#! /bin/sh -e
# unlists listmode data into span2 and creates MLrandoms estimate from the listmode data
# Author: Kris Thielemans

# right now randoms stuff only handles single time frame

# directory with some standard .par files
: ${pardir:=~/devel/STIR/examples/GE-Signa-PETMR}

# should get these parameters from command line
: ${INPUT:=LIST0000.BLF}
: ${FRAMES:=frames.fdef}

export INPUT FRAMES

# Define some example frame_duration
: ${frame_duration:=1000000000}

# Create a fdef file with some example value (replace this for your desired fdef)
if [ ! -f $FRAMES ]; then
echo "1 100000000" > $FRAMES
fi

# create prompt sinograms
OUTPUT=sinospan2 TEMPLATE=${pardir}/template.hs lm_to_projdata ${pardir}/lm_to_projdata.par 

# estimate randoms from singles
construct_randoms_from_GEsingles randomsspan2 ${INPUT} ${pardir}template.hs 

