#! /bin/bash
# $Id$

# Script to normalise simset simulations
#
# With our current phg.rec files, Simset normalises the
# output always to the same mean counts, independent
# of the number of photons simulated.
# This script multiplies the results with the number of photons
# such that they correspond to simulations of an experiment with
# a given number of counts.
#
# This script relies on names of files fixed on other scripts, so beware.

#  Copyright (C) 2005, Hammersmith Imanet Ltd
#  Copyright (C) 2011-07-01 - $Date$, Kris Thielemans
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
# Authors:  Kris Thielemans


####################### Script inputs ##################################

if [ $# -eq 0 ]; then
    echo "usage: $0 directory_name ..."
    exit 1
fi

set -e
script_name="$0"
trap "echo ERROR in script $script_name" ERR
PRINTHEADER=${SIMSET_DIR}/bin/printheader

weight_filename=`awk -F= '/weight_image_path/ '{print $2}' $1/bin.rec |tr -d \"`
all=`SimSET_STIR_names.sh $1/${weight_filename}`
while [ $# -ne 0 ]; do
  dir="$1/"
  num_photons=`grep num_to_simulate ${dir}phg.rec|awk '{print $4}'`
  for f in $all; do
    stir_math -s --including-first --times-scalar $num_photons --divide-scalar 100000000 \
      ${dir}${f}_norm.hs ${dir}${f}.hs 2> /dev/null
  done
  shift
done

