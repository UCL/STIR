#! /bin/sh -e
# unlists listmode data into span2 and creates randoms estimate from the singles in the listmode data
# Author: Kris Thielemans
# SPDX-License-Identifier: Apache-2.0
# See STIR/LICENSE.txt for details

# right now randoms stuff only handles single time frame
# we would need a simple loop to cover that case as well.

# directory with some standard .par files
: ${pardir:=$(dirname $0)}
# convert to absolute path (assumes that it exists), from DVK's answer on stackexchange
pardir=`cd "$pardir";pwd`

# should get these parameters from command line
: ${INPUT:=LIST0000.BLF}
: ${FRAMES:=frames.fdef}
: ${TEMPLATE:="$pardir"/template.hs}
export INPUT
export FRAMES
export TEMPLATE

if [ ! -f "$FRAMES" ]; then
    # make a frame definition file with 1 frame for all the data
    create_fdef_from_listmode.sh frames.fdef "$INPUT"
fi

# create prompt sinograms
OUTPUT=sinospan2 lm_to_projdata ${pardir}/lm_to_projdata.par

# estimate randoms from singles
construct_randoms_from_GEsingles randomsspan2_f1g1d0b0 "${INPUT}" sinospan2_f1g1d0b0.hs
