#! /bin/sh -e
# directory with some standard .par files
pardir=/home/kris/data/mmr

# should get these parameters from command line
INPUT=test.hlm 
FRAMES=frames.fdef 

export INPUT FRAMES

# create prompt sinograms
OUTPUT=sinospan11 TEMPLATE=${pardir}/template_span11.hs lm_to_projdata ${pardir}/lm_to_projdata.par 

# create delayed fansums
OUTPUT=fansums_delayed lm_fansums ${pardir}/lm_fansums_delayed.par 

# estimate singles from fansums
find_ML_singles_from_delayed -f MLsingles_f1 fansums_delayed_f1.dat  5 60 343 </dev/null

# estimate randoms from singles
construct_randoms_from_singles MLrandomsspan11_f1 MLsingles_f1 sinospan11_f1g1d0b0.hs 5


# example reconstruction. disabled now
if false; then
stir_math -s --mult backgroundspan11 MLrandomsspan11_f1.hs fullnormfactorsspan11.hs
export SUBSETS=14 SUBITERS=42 SAVEITERS=14
export OUTPUT=outspan11_randoms_subs${SUBSETS} 
export INPUT=sinospan11_f1g1d0b0.hs 
export SENS=subsensspan11_subs${SUBSETS}
export NORM=fullnormfactorsspan11.hs
export ADDSINO=backgroundspan11.hs
OSMAPOSL ${pardir}/OSMAPOSLbackground.par >& ${OUTPUT}.log
fi