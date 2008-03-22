#! /bin/bash
#
#
# $Id$
#
#
#  Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
# Author Charalampos Tsoumpas 
# This script is used to convert SimSET output into STIR projdata format.
# Multiple files are written for the different scatter orders in the Simset file (if present).
# Up to double scatter sinogram is currently in different files. Higher order scatter is
# currently called "multiples" which is not a good name.

# TODO: currently simset_filename and scanner_name are hard-wired in this script

if [ $# -ne 1 ]; then
    echo "usage:"
    echo "$0 max_ring_difference"
    exit 1
fi

max_ring_difference=$1
simset_file=rec.weight
scanner_name="ECAT 962"

set -e
script_name="$0"
trap "echo ERROR in script $script_name" ERR
PRINTHEADER=${SIMSET_DIR}/bin/printheader

if [ ! -r ${simset_file} ]; then
  if [ -r ${simset_file}.gz ]; then
    gunzip ${simset_file}.gz 
  else
    echo "Simset file ${simset_file} not found"
    exit 1
  fi
fi

min_scatter_bin_num=`${PRINTHEADER} ${simset_file} |grep "Binning: min number of scatters" |awk '{ print $6 }'`
max_scatter_bin_num=`${PRINTHEADER} ${simset_file} |grep "Binning: max number of scatters" |awk '{ print $6 }'`

scatter_parameter=`${PRINTHEADER} ${simset_file} |grep "Binning: scatter parameter" |awk '{ print $4 }'`

num_scatter_bins=$(( $max_scatter_bin_num - $min_scatter_bin_num + 1 ))

conv_cmdline="conv_SimSET_projdata_to_STIR \
        ${simset_file} fl \
        `${PRINTHEADER} ${simset_file} |grep "Binning: number of AA bins" |awk '{ print $6 }'` \
        `${PRINTHEADER} ${simset_file} |grep "Binning: number of TD bins" |awk '{ print $6 }'` \
        `${PRINTHEADER} ${simset_file} |grep "Binning: number of Z bins" |awk '{ print $6 }'` \
        `${PRINTHEADER} ${simset_file} |grep "Binning: max Transaxial Distance" |awk '{ print $5 }'` \
        `${PRINTHEADER} ${simset_file} |grep "Binning: range on Z value" |awk '{ print $6 }'` \
        \"${scanner_name}\" \
        $max_ring_difference "
all_scatter_bin_nums=`count $min_scatter_bin_num $max_scatter_bin_num`

case $scatter_parameter in 

  1)
  if [ $min_scatter_bin_num == 0 ]; then
    ${conv_cmdline}  0 trues
  fi
  if [ $max_scatter_bin_num == 1 ]; then
    ${conv_cmdline} 1 scatter
  fi;;


  3) # scatter_parameter

for i in $all_scatter_bin_nums ;
do for j in $all_scatter_bin_nums ;
  do 
     scatter_bin_num=$(( ${num_scatter_bins} * $i + $j ))
     echo extracting scatter bin $scatter_bin_num
     echo ${conv_cmdline} ${scatter_bin_num} blue${i}_pink${j};
     ${conv_cmdline} ${scatter_bin_num} blue${i}_pink${j};
  done 
done 

# get list of all contributing sinograms from blue/pink sinograms

# first initialise lists to empty
all_trues=
all_singles=
all_singles12=
all_doubles1=
all_doubles2=
all_multiples=
for i in $all_scatter_bin_nums
  do for j in $all_scatter_bin_nums
    do     
      k=$(( $i + $j )) 
      current=blue${i}_pink${j}.hs
      if [ $k -eq 0 ]; then all_trues="$all_trues $current"; fi
      if [ $k -eq 1 ]; then all_singles="$all_singles $current"; fi
      if [ $k -eq 2 ]; then  if [ $i -eq 1 ]; then all_singles12="$all_singles12 $current";fi; fi;
      if [ $k -eq 2 ]; then  if [ $j -eq 2 ]; then all_doubles1="$all_doubles1 $current";fi; fi;    
      if [ $k -eq 2 ]; then  if [ $i -eq 2 ]; then all_doubles2="$all_doubles2 $current";fi; fi;
      if [ $k -gt 2 ]; then all_multiples="$all_multiples $current"; fi      
    done
done

# TODO we currently pipe all output to /dev/null because there's a lot of stuff because blocksizes do not work
stir_math -s --add trues $all_trues >& /dev/null
stir_math -s --add singles $all_singles >& /dev/null
if [ ! -z "$all_singles12" ]; then
  stir_math -s --add doubles11 $all_singles12 >& /dev/null
fi
if [ ! -z "$all_doubles1" ]; then
  stir_math -s --add doubles20 $all_doubles1 $all_doubles2 >& /dev/null
fi
stir_math -s --add multiples $all_multiples >& /dev/null

rm -f blue*s
  #end of  scatter_param == 3
  ;;  

  *)
    echo "scatter_parameter $scatter_parameter currently unsupported"
    exit 1
    ;;

esac
