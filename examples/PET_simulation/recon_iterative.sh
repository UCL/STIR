#! /bin/sh

# A simple script to run OSMAPOSL or OSSPS with "corrections" as input.
# This script is part of the PET simulation example and assumes that the reconstruction .par file 
# uses INPUT, OUTPUT, MULTFACTORS and ADDSINO variables.
#
# If you have installed STIR with MPI, you will need to first set the MPIRUN
# environment variable. In sh/bash/ksh etc, the following might work
#
# MPIRUN="mpirun -np 4"
# export MPIRUN

#  Copyright (C) 2013-2014 University College London
#  This file is part of STIR.
#
#  SPDX-License-Identifier: Apache-2.0
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
  INPUT=${prompts} OUTPUT=${output} MULTFACTORS=my_norm.hs ADDSINO=my_zeros.hs ${MPIRUN} ${reconProg} ${parfile}
else
  INPUT=${prompts} OUTPUT=${output} MULTFACTORS=${multfactors} ADDSINO=${addsino} ${MPIRUN} ${reconProg} ${parfile}
fi
if [ $? -ne 0 ]; then 
  echo "ERROR running iterative recon"; exit 1; 
fi
echo "${reconProg} done"

