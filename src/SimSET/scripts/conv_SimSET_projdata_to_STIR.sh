#! /bin/bash
#
# This script is used to convert SimSET output (weight file) into STIR projdata format.
# Multiple files are written for the different scatter orders in the Simset file (if present).
# The names of these files are currently not very predictable. Use SimSET_STIR_names.sh to
# find out what we used.
#
# Most settings of SimSET's scatter_parameter are supported but not all. If you need
# others, you can modify this script and contribute it to the STIR list.
#
# If SimSET's scatter_parameter was set to 3, sinograms for 2nd order scatter are added 
# together (doubles02 and doubles11). Even higher order scatter is summed into a file
# currently called "multiples" which is not a good name.
# However, since SimSET 2.9, you probably want to use scatter_parameter 4 or 5 instead.
#
# Names of the output files do not reflect if randoms are included or not. 
# If you use scatter_parameter>=6, the total randoms are stored in a separate 
# file (called randoms) in most cases.
#
# WARNING: If the scanner name is not supported by STIR, some information 
# (in particular the ring diameter and average DOI) will be incorrect in the
# *.hs files. You will need to correct this by hand for the moment.

#
#  Copyright (C) 2005 - 2008/03/22, Hammersmith Imanet Ltd
#  Copyright (C) 2011-07-01 - 2012, Kris Thielemans
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


if [ $# -ne 3 ]; then
    echo "usage:"
    echo "$0 simset-weight-file max_ring_difference stir-scanner-name"
    echo "Surround the scanner name with quotes if it contains spaces"
    exit 1
fi

simset_file=$1
max_ring_difference=$2
scanner_name=$3

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

# find min and max scatter bins in the SimSET file
min_scatter_bin_num=`${PRINTHEADER} ${simset_file} |grep "Binning: min number of scatters" |awk '{ print $6 }'`
max_scatter_bin_num=`${PRINTHEADER} ${simset_file} |grep "Binning: max number of scatters" |awk '{ print $6 }'`

# find scatter_parameter (maybe called scatter_randoms_parameter in the future)
scatter_parameter=`${PRINTHEADER} ${simset_file} |grep "Binning: scatter.*parameter" |awk '{ print $4 }'`

num_scatter_bins=$(( $max_scatter_bin_num - $min_scatter_bin_num + 1 ))

# construct main part of the command line for the conv_SimSET_projdata_to_STIR executable
# by parsing the header of the SimSET file
conv_cmdline="conv_SimSET_projdata_to_STIR \
        ${simset_file} fl \
        `${PRINTHEADER} ${simset_file} |grep "Binning: number of AA bins" |awk '{ print $6 }'` \
        `${PRINTHEADER} ${simset_file} |grep "Binning: number of TD bins" |awk '{ print $6 }'` \
        `${PRINTHEADER} ${simset_file} |grep "Binning: number of Z bins" |awk '{ print $6 }'` \
        `${PRINTHEADER} ${simset_file} |grep "Binning: max Transaxial Distance" |awk '{ print $5 }'` \
        `${PRINTHEADER} ${simset_file} |grep "Binning: range on Z value" |awk '{ print $6 }'` \
        '${scanner_name}' \
        $max_ring_difference "
all_scatter_bin_nums=`count $min_scatter_bin_num $max_scatter_bin_num`

echo Executing ${conv_cmdline}

case $scatter_parameter in 

  0)
    eval ${conv_cmdline}  0 total
    ;;

  1|6)
    if [ $min_scatter_bin_num == 0 ]; then
      eval ${conv_cmdline}  0 noscatter
      if [ $max_scatter_bin_num -ge 1 ]; then
        eval ${conv_cmdline} 1 scatter
        if [ $scatter_parameter == 6 ]; then
	  eval ${conv_cmdline} 2 randoms
        fi
      else
        if [ $scatter_parameter == 6 ]; then
          eval ${conv_cmdline} 1 randoms
        fi
      fi
    fi
   ;;


  3|8) # scatter_parameter

    for i in $all_scatter_bin_nums ;
    do for j in $all_scatter_bin_nums ;
      do 
         scatter_bin_num=$(( ${num_scatter_bins} * $i + $j ))
         echo extracting scatter bin $scatter_bin_num
         echo ${conv_cmdline} ${scatter_bin_num} blue${i}_pink${j};
         eval ${conv_cmdline} ${scatter_bin_num} blue${i}_pink${j};
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

    if [ $scatter_parameter == 8 ]; then
      echo "WARNING: Not storing randoms yet"
    fi
    #end of  scatter_param == 3
    ;;  

  4|5|9|10) # scatter_parameter
    # (PET-only) to histogram coincidences into one index with (max_s-min_s)+1 bins 
    # using (blueScatters + pinkScatters) to compute the index number. 
    # The coincidence is rejected if the sum is < min_s (or > max_s for scatter_param 4,9).

    for scatter_bin_num in $all_scatter_bin_nums ;
    do 
        echo extracting scatter bin $scatter_bin_num
        idx=$((scatter_bin_num - min_scatter_bin_num))
        echo ${conv_cmdline} ${idx} scatter${scatter_bin_num};
        eval ${conv_cmdline} ${idx} scatter${scatter_bin_num};
    done
    if [ $scatter_parameter == 10 -o $scatter_parameter == 9  ]; then
        echo ${conv_cmdline} $((max_scatter_bin_num - min_scatter_bin_num + 1)) randoms
        eval ${conv_cmdline} $((max_scatter_bin_num - min_scatter_bin_num + 1)) randoms
    fi 
    ;;

  *)
    echo "scatter_parameter $scatter_parameter currently unsupported"
    exit 1
    ;;

esac
