#! /bin/sh
# A simple script to run FBP2D with precorrections.
# This script is part of the PET simulation example and assumes that FBP .par file 
# uses INPUT and OUTPUT variables.

#  Copyright (C) 2013-2014 University College London
#  This file is part of STIR.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  See STIR/LICENSE.txt for details
#      
#  Author Kris Thielemans

if [ $# -ne 3 -a $# -ne 5 ]; then
  echo "Usage: `basename $0` output_prefix prompts FBPparfile [ multfactors additive_sinogram ]"
  exit 1
fi

output=$1
prompts=$2
parfile=$3
multfactors=$4
addsino=$5

if [ $# -eq 3 ]; then
  # no corrections
  INPUT=${prompts} OUTPUT=${output} FBP2D ${parfile}  > ${output}.log 2>&1
  if [ $? -ne 0 ]; then 
    echo "ERROR running FBP2D. Check ${output}.log"; exit 1; 
  fi
else
  # do precorrections
  # normalise (and correct for attenuation if its in multfactors)
  stir_math -s --mult ${output}_precorrected.hs  ${prompts} ${multfactors}
  if [ $? -ne 0 ]; then 
    echo "ERROR running stir_math for norm"; exit 1; 
  fi
  #remove  background
  stir_subtract -s --accumulate ${output}_precorrected.hs ${addsino}
  if [ $? -ne 0 ]; then 
    echo "ERROR running stir_math for background subtraction"; exit 1; 
  fi
  # reconstruct
  INPUT=${output}_precorrected.hs OUTPUT=${output} FBP2D ${parfile}  > ${output}.log 2>&1
  if [ $? -ne 0 ]; then 
    echo "ERROR running FBP2D. Check ${output}.log"; exit 1; 
  fi

fi
echo "FBP2D done"
