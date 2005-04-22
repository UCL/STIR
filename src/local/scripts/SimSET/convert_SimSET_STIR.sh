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

min_scatter_bin_num=`~/simset/bin/printheader rec.weight |grep "Binning: min number of scatters" |awk '{ print $6 }'`
max_scatter_bin_num=`~/simset/bin/printheader rec.weight |grep "Binning: max number of scatters" |awk '{ print $6 }'`

num_scatter_bins=$(( $max_scatter_bin_num - $min_scatter_bin_num + 1 ))

all_scatter_bin_nums=`count $min_scatter_bin_num $max_scatter_bin_num`

for i in $all_scatter_bin_nums ;
do for j in $all_scatter_bin_nums ;
do 
 scatter_bin_num=$(( ${num_scatter_bins} * $i + $j ))
 echo extracting scatter bin $scatter_bin_num

~/parapet/PPhead/opt/local/SimSET/conv_SimSET_STIR \
rec.weight fl \
`~/simset/bin/printheader rec.weight |grep "Binning: number of AA bins" |awk '{ print $6 }'` \
`~/simset/bin/printheader rec.weight |grep "Binning: number of TD bins" |awk '{ print $6 }'` \
`~/simset/bin/printheader rec.weight |grep "Binning: number of Z bins" |awk '{ print $6 }'` $max_ring_difference \
`~/simset/bin/printheader rec.weight |grep "Binning: max Transaxial Distance" |awk '{ print $5 }'` \
${scatter_bin_num}  blue${i}_pink${j} ;
done ; done 

# get list of all contributing sinograms from blue/pink sinograms

# first initialise lists to empty
all_trues=
all_singles=
all_doubles=
all_multiples=
for i in $all_scatter_bin_nums;
  do for j in $all_scatter_bin_nums;
  do
  current=blue${i}_pink${j}.hs
  case $(( $i + $j )) in
    0) all_trues="$all_trues $current";;
    1) all_singles="$all_singles $current";;
    2) all_doubles="$all_doubles $current";;
    *) all_multiples="$all_multiples $current";;
  esac
  done
done

# TODO we currently pipe all output to /dev/null because there's a lot of stuff because blocksizes do not work
stir_math -s --add trues $all_trues >& /dev/null
stir_math -s --add singles $all_singles >& /dev/null
stir_math -s --add doubles $all_doubles >& /dev/null
stir_math -s --add multiples $all_multiples >& /dev/null

rm -f blue*s


