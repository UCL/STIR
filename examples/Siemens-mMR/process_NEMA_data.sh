#! /bin/sh -e
# Example script to unlist and reconstruct NEMA data from https://zenodo.org/record/1304454
# Author: Kris Thielemans

# default location of directory with parameter files
: ${pardir:=~/devel/STIR/examples/Siemens-mMR}
export pardir

### create projection data (sinograms) from listmode file
# use 500s of data
echo "1 500" > frames.fdef
#INPUT=20170809_NEMA_60min_UCL.l.hdr FRAMES=frames.fdef $pardir/unlist_and_randoms.sh

atnimg=20170809_NEMA_MUMAP_UCL.v.hdr ECATNORM=20170809_NEMA_UCL.n.hdr $pardir/scatter_and_recon.sh
