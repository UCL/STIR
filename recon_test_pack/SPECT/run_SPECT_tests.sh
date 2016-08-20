#! /bin/sh
# A script to check to see if reconstruction of simulated data gives the expected result.
#
#  Copyright (C) 2011, Hammersmith Imanet Ltd
#  Copyright (C) 2014, University College London
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
# 

# Scripts should exit with error code when a test fails:

if [ -n "$TRAVIS" ]; then
    # The code runs inside Travis
    set -e
fi

echo This script should work with STIR version 3.0. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.
echo

#
# Options
#
MPIRUN=""
CACHEALLVIEWS2D=0
# always disable the cache for 3D PSF as we run less than 1 iteration
CACHEALLVIEWS3D=0
export CACHEALLVIEWS2D
export CACHEALLVIEWS3D

#
# Parse option arguments (--)
# Note that the -- is required to suppress interpretation of $1 as options 
# to expr
#
while test `expr -- "$1" : "--.*"` -gt 0
do

  if test "$1" = "--mpicmd"
  then
    MPIRUN="$2"
    shift 1
  elif test "$1" = "--usecache"
  then
    CACHEALLVIEWS2D=1
  elif test "$1" = "--help"
  then
    echo "Usage: `basename $0` [--mpicmd somecmd] [--usecache] [install_dir]"
    echo "(where [] means that an argument is optional)"
    echo "See README.txt for more info."
    exit 1
  else
    echo Warning: Unknown option "$1"
    echo rerun with --help for more info.
    exit 1
  fi

  shift 1

done 

if [ $# -eq 1 ]; then
  echo "Prepending $1 to your PATH for the duration of this script."
  PATH=$1:$PATH
fi

if [ ${CACHEALLVIEWS2D} -eq 1 ]; then
   echo "Keeping all views in memory for the iterative reconstruction."
else
   echo "Recomputing the matrix in every iteration for the iterative reconstruction."
   echo "If you have plenty of memory, relaunch with the option \"--usecache\""
fi

command -v OSMAPOSL >/dev/null 2>&1 || { echo "OSMAPOSL not found or not executable. Aborting." >&2; exit 1; }
echo "Using `command -v FBP2D`"
echo "Using `command -v OSMAPOSL`"

# first need to set this to the C locale, as this is what the STIR utilities use
# otherwise, awk might interpret floating point numbers incorrectly
LC_ALL=C
export LC_ALL

rm -rf out
mkdir out

# loop over reconstruction algorithms
error_log_files=""

for reconpar in FBP2D OSEM_2DPSF OSEM_3DPSF; do
    # test first if analytic reconstruction and if so, run pre-correction
    isFBP=0
    if expr ${reconpar} : FBP > /dev/null; then
      isFBP=1
      recon=FBP2D
    else
      isFBP=0
      recon=OSMAPOSL
    fi

    echo "============================================="
    #echo "Using `command -v ${recon}`"

    parfile=${reconpar}.par
    # run actual reconstruction
    echo "Running ${recon} ${parfile}"
    logfile=out/${parfile}.log
    ${MPIRUN} ${recon} ${parfile} > ${logfile} 2>&1
    if [ $? -ne 0 ]; then
       echo "Error running reconstruction. CHECK RECONSTRUCTION LOG ${logfile}"
       error_log_files="${error_log_files} ${logfile}"
       break
    fi

    # find filename of (last) image from ${parfile}
    output_filename=`awk -F':='  '/output[ _]*filename[ _]*prefix/ { value=$2;gsub(/[ \t]/, "", value); printf("%s", value) }' ${parfile}`
    if [ ${isFBP} -eq 0 ]; then
      # iterative algorithm, so we need to append the num_subiterations
      num_subiterations=`awk -F':='  '/number[ _]*of[ _]*subiterations/ { value=$2;gsub(/[ \t]/, "", value); printf("%s", value) }' ${parfile}`
      output_filename=${output_filename}_${num_subiterations}
    fi
    output_image=${output_filename}.hv

    # horrible way to replace "out" with "org" (as we don't want to rely on bash)
    org_output_image=out`echo ${output_image}|cut -c 4-`

    if compare_image ${org_output_image} ${output_image}
    then
    echo ---- This test seems to be ok !;
    else
    echo There were problems here!;
    error_log_files="${error_log_files} my_${parfile}.log"
    fi
done

echo "============================================="
if [ -z "${error_log_files}" ]; then
 echo "All tests OK!"
 echo "You can remove all output using \"rm -rf out\""
else
 echo "There were errors. Check ${error_log_files}"
fi

