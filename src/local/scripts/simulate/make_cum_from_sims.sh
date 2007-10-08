#! /bin/bash
#PBS -k eo 
#PBS -l vmem=200mb
# Author: Kris Thielemans
# Author: Charalampos Tsoumpas
# A script which add the different seed parametric reconstructions as in ../../simulate/combine*

print_usage_and_exit()
{
  echo "usage: "
  echo "  $PROG simname prefix suffix " 1>&2
  echo ' Looks for files ${prefix}${sim}${suffix}'
  echo " currently uses env_variables allsims and ADDTOPREVIOUS"
  echo " simname cannot contain path-separators. files will be written as cum_\${simname}"
  exit 1
}


# find script name (i.e. get rid of directory part)
PROG=${0##*/}


# option processing
#if [ $# -gt 0 ]; then
#  while [ "${1:0:1}" = "-"  ]; do  # while 1st arg starts with -
#    case "$1" in 
#    --rdf) rdf_file=$2; fromRDF=1; shift;shift;;
#    *) print_usage_and_exit;;
#    esac
#  done
#fi



if [ $# != 3 ]; then
  print_usage_and_exit
fi

set -e
trap " echo ERROR in executing script" ERR


if [ -z "$ADDTOPREVIOUS" ]; then 
  ADDTOPREVIOUS=0
fi
#echo addtoprevious $ADDTOPREVIOUS
simname=$1
prefix=$2
suffix=$3

# TODO add --parameter and change extension

extension=hv

function make_current {
  current=${prefix}${sim}${suffix}
}


if [ "$ADDTOPREVIOUS" -eq "0" ]; then
   num_trials=0
   rm -f ${simname}_trials ${simname}_current_trial 
   # find first file and use as a template to get a 0 file
   already_found_file=0
   for sim in $allsims; do
     if [ $already_found_file = 1 ]; then continue; fi
     make_current $sim
     if [ -r $current ]; then
        for pow in 1 2 3 4; do	  
          stir_math --including-first --times-scalar 0 cum_${simname}_pow${pow}.${extension} $current
        done
	already_found_file=1
     fi   
   done 
else
  if [ ! -r cum_${simname}_pow1.${extension} ]; then
    echo "You do not have any previous image to add. "
    exit 1
  else
    numtrials=`cat ${simname}_current_trial`
  fi
fi

rm -f ${simname}_trials_to_add
for sim in $allsims; do
  # find out if sim is already in ${simname}_trials
  if [ -r  ${simname}_trials ]; then
    if [ `awk  "BEGIN {n=0} /^${sim}\$/ { n=n+1} END {print n}"  ${simname}_trials` = 1 ]; then
      # sim already processed
      continue
    fi
  fi

  make_current $sim

  if [ ! -r $current ]; then
    echo $current does not exist. skipping
    continue
  fi
  
  #echo Adding $current
  echo $current >> ${simname}_trials_to_add
  numtrials=`expr $numtrials + 1`
  echo $sim >> ${simname}_trials
  
done

for pow in 1 2 3 4; do
  cat ${simname}_trials_to_add | xargs stir_math --power $pow --accumulate cum_${simname}_pow${pow}.${extension} 
done

rm -f ${simname}_trials_to_add
rm -f ${simname}_current_trial
echo $numtrials > ${simname}_current_trial
