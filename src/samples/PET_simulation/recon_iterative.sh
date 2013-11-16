#! /bin/sh
# call OSMAPOSL for our simulated data
if [ $# -lt 5 ]; then
  echo "Usage: `basename $0` output_prefix prompts reconProg OSMAPOSLparfile multfactors additive_sinogram "
  exit 1
fi

output=$1
prompts=$2
reconProg=$3
parfile=$4
multfactors=$5
addsino=$6


if [ $# -eq 4 ]; then
  # no background
  INPUT=${prompts} OUTPUT=${output} MULTFACTORS=my_norm.hs ADDSINO=my_zeros.hs ${reconProg} ${parfile}
else
  INPUT=${prompts} OUTPUT=${output} MULTFACTORS=${multfactors} ADDSINO=${addsino} ${reconProg} ${parfile}
fi
if [ $? -ne 0 ]; then 
  echo "ERROR running iterative recon"; exit 1; 
fi
echo "${reconProg} done"

