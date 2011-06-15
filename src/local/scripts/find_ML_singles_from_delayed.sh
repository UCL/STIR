#! /bin/bash
# file to provide somewhat friendlier interface to find_ML_singles_from_delayed.
# can also write csv files for import into e.g. matlab

# Author: Kris Thielemans
# $Id$

#defaults
makeCSV=1
niter=30
KLInterval=""
saveInterval=""

#save for later
cmd_args=$*


# option processing

while [ 1 -eq 1 ];
do
  case $1 in 
   --KLInterval) KLInterval=$2; shift;shift;;
   --systemType) system_type=$2; shift;shift;;
   --saveInterval) saveInterval=$2; shift;shift;;
   --numIterations) niter=$2; shift;shift;;
   --makeCSV) makeCSV=$2; shift;shift;;
   -h | --help)
     cat <<EOF
Interface to compute ML singles from delayeds. Just type without argument for usage.
Optionally makes cvs files (although dangerously as it assumes that the files
were written by STIR as it isn't careful with line breaks etc
EOF
    exit 1;;

   *) break;;
  esac
done

cmdline_ok=0
if [ $# -eq 2 ]; then
  if [ -r $2 ]; then
   cmdline_ok=1
  else
   echo "Error: last argument does not correspond to a readable file" 1>&2
   exit 1
  fi
fi
if [ $cmdline_ok -eq 0 ]; then
    echo "usage: $0 \\"
    echo "  [--numIterations n] [--saveInterval s] [--KLInterval k ] \\"
    echo "  [--makeCSV 0|1 ] --systemType 966|962 outputPrefix FansumFile"
    exit 1
fi

fansumfile=$2
output=$1

# ok, finally do some work

if [ "${system_type}" -eq "966" ]; then
    num_rings=48
    fan_size=287
elif [ "${system_type}" -eq "962" ]; then
    num_rings=32
    fan_size=287
else
    echo "ERROR: Need to know num_rings for randoms estimation, but unknown system_type ${system_type}" 1>&2
    exit 1
fi

if true; then
find_ML_singles_from_delayed -f    $output $fansumfile ${niter} $num_rings $fan_size <<EOF > $output.log
0
$KLInterval
$saveInterval
EOF
fi

if [ $makeCSV == 1 ]; then
    for iter in `count $saveInterval $niter  $saveInterval`; do
        rm -f ${output}_eff_1_${iter}.csv 
        sed -e 's/}//' -e 's/, {//'  -e 's/{{//' <${output}_eff_1_${iter}.out >  ${output}_eff_1_${iter}.csv;
    done
    rm -f ${output}_eff_1_${niter}.csv ${fansumfile%.*}.csv
    sed -e 's/}//' -e 's/, {//'  -e 's/{{//' <${fansumfile} >  ${fansumfile%.*}.csv

fi
