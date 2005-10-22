#
# $Id$
#
#
#  Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
#      See STIR/LICENSE.txt for details
#      
# Author /Kris Thielemans
# Author /Charalampos Tsoumpas 
# This script is used to convert SimSET output into STIR projdata format, splitting double scatter sinogram into different files.

#! /bin/bash
#
if [ $# -ne 1 ]; then
    echo "usage:"
    echo "$0 max_ring_difference"
    exit 1
fi

max_ring_difference=$1

set -e
trap "echo ERROR in script " ERR

#echo dir $DIR
#cd ~/sim_pet/data/${DIR}

if [ ! -r rec.weight ]; then
gunzip rec.weight.gz 
fi

min_scatter_bin_num=`${SIMSET_DIR}/bin/printheader rec.weight |grep "Binning: min number of scatters" |awk '{ print $6 }'`
max_scatter_bin_num=`${SIMSET_DIR}/bin/printheader rec.weight |grep "Binning: max number of scatters" |awk '{ print $6 }'`

num_scatter_bins=$(( $max_scatter_bin_num - $min_scatter_bin_num + 1 ))

all_scatter_bin_nums=`count $min_scatter_bin_num $max_scatter_bin_num`

for i in $all_scatter_bin_nums ;
do for j in $all_scatter_bin_nums ;
  do 
     scatter_bin_num=$(( ${num_scatter_bins} * $i + $j ))
     echo extracting scatter bin $scatter_bin_num

     conv_SimSET_STIR \
        rec.weight fl \
        `${SIMSET_DIR}/bin/printheader rec.weight |grep "Binning: number of AA bins" |awk '{ print $6 }'` \
        `${SIMSET_DIR}/bin/printheader rec.weight |grep "Binning: number of TD bins" |awk '{ print $6 }'` \
        `${SIMSET_DIR}/bin/printheader rec.weight |grep "Binning: number of Z bins" |awk '{ print $6 }'` \
        $max_ring_difference \
        `${SIMSET_DIR}/bin/printheader rec.weight |grep "Binning: max Transaxial Distance" |awk '{ print $5 }'` \
        ${scatter_bin_num} blue${i}_pink${j};
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
stir_math -s --add doubles11 $all_singles12 >& /dev/null
stir_math -s --add doubles20 $all_doubles1 $all_doubles2 >& /dev/null
stir_math -s --add multiples $all_multiples >& /dev/null

rm -f blue*s


