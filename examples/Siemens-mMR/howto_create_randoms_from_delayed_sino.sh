#! /bin/sh -e
# given delayed sino, output ML-estimate of randoms in the same "shape" as a template
# input should be in span 1. template can be any span
# Example (in bash or sh):
# INPUT=dsino_span1.hs TEMPLATE=psino_span11.hs howto_create_randoms_from_delayed_sino.sh 

# Author: Kris Thielemans

# TODO: should get these parameters from command line
: ${INPUT:=dsino_span1.hs}
: ${TEMPLATE:=${INPUT}}
: ${NUMITERS:=10}

# estimate singles from fansums
find_ML_singles_from_delayed MLsingles ${INPUT}  ${NUMITERS} </dev/null

# estimate randoms from singles
construct_randoms_from_singles MLrandoms MLsingles ${TEMPLATE} ${NUMITERS}

