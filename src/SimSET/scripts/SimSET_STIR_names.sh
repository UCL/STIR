#! /bin/bash
#
#
# $Id$
#
#
#  Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
#      See STIR/LICENSE.txt for details
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

scatter_parameter=`${PRINTHEADER} ${simset_file} |grep "Binning: scatter parameter" |awk '{ print $4 }'`

all=""
case $scatter_parameter in 

  1)
  if [ $min_scatter_bin_num == 0 ]; then
    all="$all trues"
  fi
  if [ $max_scatter_bin_num == 1 ]; then
     all="$all scatter"
  fi
  ;;


  3) # scatter_parameter

  all="doubles11  doubles20  multiples  singles  trues"
  ;;

  *)
  # unsupported
  ;;
esac
echo $all

