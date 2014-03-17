#! /bin/sh

# A simple script to run OSMAPOSL or OSSPS with "corrections" as input.
# This script is part of the PET simulation example and assumes that the reconstruction .par file 
# uses INPUT, OUTPUT, MULTFACTORS and ADDSINO variables.

#  Copyright (C) 2013-2014 University College London
#  This file is part of STIR.
#
#  This file is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 2.1 of the License, or
#  (at your option) any later version.

#  This file is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  See STIR/LICENSE.txt for details
#      
#  Author Kris Thielemans

if [ $# -ne 6 ]; then
  echo "Usage: `basename $0` output_prefix prompts reconProg reconParfile multfactors additive_sinogram "
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

