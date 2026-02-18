#! /bin/bash
#
#  This scripts adds STIR projection data as produced by the SimSET conversion tool.
#  WARNING: it assumes a particular scaling of the SimSET simulation, which might
#  depend on the phg input file (see mult_num_photons.sh).
#
#  Copyright (C) 2005, Hammersmith Imanet Ltd
#  Copyright (C) 2011-07-01 - 2012, Kris Thielemans
#  This file is part of STIR.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  See STIR/LICENSE.txt for details
#      
# Authors:  Kris Thielemans

prog=$0
if [ $# -lt 2 ]; then
    echo "usage:"
    echo "$0 out_dir input_dir1 input_dir2 ...."
    exit 1
fi

set -e
trap "echo ERROR in script $prog" ERR

out_dir=$1
mkdir -p $out_dir

shift
echo "will process $*"


mult_num_photons.sh $*

weight_filename=`awk -F= '/weight_image_path/ '{print $2}' $1/bin.rec |tr -d \"`
all=(`SimSET_STIR_names.sh $1/${weight_filename}`)


for i in `count 0 $(( ${#all[*]} - 1))`; do
  all_files=""
  for dir in $*; do
     all_files="${all_files} $dir/${all[i]}_norm.hs"
  done
  echo "Adding ${all_files}"  
# TODO we currently pipe all output to /dev/null because there's a lot of stuff because blocksizes do not work
trap "echo ERROR in executing stir_math  $out_dir/${all[i]}  ${all_files} " ERR
  stir_math -s --add $out_dir/${all[i]}  ${all_files} >& /dev/null
done

