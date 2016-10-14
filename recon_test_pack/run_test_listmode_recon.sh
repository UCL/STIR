#! /bin/sh
# A script to check to see if reconstruction of simulated data gives the expected result.
#
#  Copyright (C) 2011 - 2011-01-14, Hammersmith Imanet Ltd
#  Copyright (C) 2011-07-01 - 2011, Kris Thielemans
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
# Author Nikos Efthimiou
#

# Scripts should exit with error code when a test fails:
if [ -n "$TRAVIS" ]; then
    # The code runs inside Travis
    set -e
fi

echo This script should work with STIR version ">"3.0. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.
echo

#
# Options
#
MPIRUN=""

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
  elif test "$1" = "--help"
  then
    echo "Usage: `basename $0` [--mpicmd somecmd] [install_dir]"
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

# first delete any files remaining from a previous run
rm -f my_*v my_*s my_*S

echo "===  reconstruct listmode data"
OSMAPOSL OSMAPOSL_test_lmf.par
echo "===  "
# create sinograms
echo "===  unlist listmode data (for comparison)"
INPUT=PET_ACQ_small.l.hdr.STIR TEMPLATE=Siemens_mMR_seg2.hs OUT_PROJDATA_FILE=my_sinogram lm_to_projdata  lm_to_projdata.par
echo "===  reconstruct projection data for comparison"
OSMAPOSL OSMAPOSL_test_proj.par
echo "=== compare sensitivity images"
if ${INSTALL_DIR}compare_image my_sens_t_proj_seg2.hv my_sens_t_lm_pr_seg2.hv 2>my_sens_comparison_stderr.log;
then
echo ---- This test seems to be ok !;
else
echo There were problems here!;
ThereWereErrors=1;
fi

echo "=== compare reconstructed images"
if ${INSTALL_DIR}compare_image my_output_t_proj_seg2_1.hv my_output_t_lm_pr_seg2_1.hv 2>my_output_comparison_stderr.log;
then
echo ---- This test seems to be ok !;
else
echo There were problems here!;
ThereWereErrors=1;
fi

echo
echo '--------------- End of tests -------------'
echo
if test ${ThereWereErrors} = 1  ;
then
echo "Check what went wrong. The *.log files might help you."
else
echo "Everything seems to be fine !"
echo 'You could remove all generated files using "rm -f my_* *.log"'
fi

