#! /bin/sh -e
# unlists listmode data into span11 and creates MLrandoms estimate from the listmode data
# Author: Kris Thielemans

# right now randoms stuff only handles single time frame

# directory with some standard .par files
: ${pardir:=~/devel/STIR-UCL-extensions/mMR}

# should get these parameters from command line
: ${INPUT:=test.hlm}
: ${FRAMES:=frames.fdef}

export INPUT FRAMES

# create prompt sinograms
OUTPUT=sinospan11 TEMPLATE=${pardir}/template_span11.hs lm_to_projdata ${pardir}/lm_to_projdata.par 

# create delayed fansums
OUTPUT=fansums_delayed lm_fansums ${pardir}/lm_fansums_delayed.par 

# estimate singles from fansums
find_ML_singles_from_delayed -f MLsingles_f1 fansums_delayed_f1.dat  5 60 343 </dev/null

# estimate randoms from singles
construct_randoms_from_singles MLrandomsspan11_f1 MLsingles_f1 sinospan11_f1g1d0b0.hs 5
