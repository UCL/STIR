#! /bin/bash
#
if [ $# -lt 2 ]; then
    echo "usage:"
    echo "$0 out_dir input_dir1 input_dir2 ...."
    exit 1
fi

out_dir=$1
shift
echo "will process $*"
set -e
trap "echo ERROR in script " ERR

mult_num_photons.sh $*

all=(doubles11  doubles20  multiples  singles  trues)

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

