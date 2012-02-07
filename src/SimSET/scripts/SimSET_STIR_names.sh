#! /bin/bash
# This scripts prints the names of the projection data constructed by
# conv_SimSET_projdata_to_STIR.sh when converting a Simset weight file.
#
# $Id$
#
#
#  Copyright (C) 2005 - 2005-10-24, Hammersmith Imanet Ltd
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
# Author Kris Thielemans


if [ $# -ne 1 ]; then
    echo "usage:"
    echo "$0 simset_file"
    exit 1
fi

script_name="$0"
trap "echo ERROR in script $script_name" ERR
PRINTHEADER=${SIMSET_DIR}/bin/printheader

simset_file="$1"
min_scatter_bin_num=`${PRINTHEADER} ${simset_file} |grep "Binning: min number of scatters" |awk '{ print $6 }'`
max_scatter_bin_num=`${PRINTHEADER} ${simset_file} |grep "Binning: max number of scatters" |awk '{ print $6 }'`
all_scatter_bin_nums=`count $min_scatter_bin_num $max_scatter_bin_num`

scatter_parameter=`${PRINTHEADER} ${simset_file} |grep "Binning: scatter.*parameter" |awk '{ print $4 }'`

all=""
case $scatter_parameter in 

  0)
  all="total"
  ;;
  1|6)
  all=""  
  if [ $min_scatter_bin_num == 0 ]; then
    all="$all noscatter"
    if [ $max_scatter_bin_num -ge 1 ]; then
      all="$all scatter"
      if [ $scatter_parameter == 6 ]; then
	  all="$all randoms"
      fi
    else
      if [ $scatter_parameter == 6 ]; then
	  all="$all randoms"
      fi
    fi
  fi
  ;;


  3|8) # scatter_parameter

  all="doubles11  doubles20  multiples  singles  trues"
  ;;

  4|5|9|10)
  all=""
  for i in $all_scatter_bin_nums ;
  do 
      all="$all_bin_num} scatter${i};"
  done
  if [ $scatter_parameter == 10 -o $scatter_parameter == 9  ]; then
      all="$all_scatter_bin_num - min_scatter_bin_num + 1)) randoms"
  fi 
  ;;

  *)
  # unsupported
  ;;
esac
echo $all

