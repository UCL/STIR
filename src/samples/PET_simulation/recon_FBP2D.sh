#! /bin/sh

if [ $# -lt 3 ]; then
  echo "Usage: `basename $0` output_prefix prompts FBPparfile multfactors additive_sinogram "
  exit 1
fi

output=$1
prompts=$2
parfile=$3
multfactors=$4
addsino=$5


if [ $# -eq 3 ]; then
  # no corrections
  INPUT=my_prompts.hs OUTPUT=${output} FBP2D ${parfile}  > my_fbp.log 2>&1
else
  # do precorrections
  # normalise
  stir_math -s --mult my_precorrected.hs  ${prompts} ${multfactors}
  if [ $? -ne 0 ]; then 
    echo "ERROR running stir_math for norm"; exit 1; 
  fi
  #remove  background
  stir_subtract -s --accumulate my_precorrected.hs ${addsino}
  if [ $? -ne 0 ]; then 
    echo "ERROR running stir_math for background subtraction"; exit 1; 
  fi
  # reconstruct
  INPUT=my_precorrected.hs OUTPUT=${output} FBP2D ${parfile}  > ${output}.log 2>&1
  if [ $? -ne 0 ]; then 
    echo "ERROR running FBP2D"; exit 1; 
  fi

fi
echo "FBP2D done"