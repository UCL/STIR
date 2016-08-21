#! /bin/sh
# Shell script for automatic running of the tests
# see README.txt
#  Copyright (C) 2000 - 2001 PARAPET partners
#  Copyright (C) 2001 - 2009-10-11, Hammersmith Imanet Ltd
#  Copyright (C) 2011, Kris Thielemans
#  Copyright (C) 2013 - 2014, University College London
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

# Scripts should exit with error code when a test fails:
if [ -n "$TRAVIS" ]; then
    # The code runs inside Travis
    set -e
fi

echo This script should work with STIR version 2.1, 2.2, 2.3, 2.4 and 3.0. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.
echo

#
# Options
#
NOINTBP=0

MPIRUN=""

#
# Parse option arguments (--)
# Note that the -- is required to suppress interpretation of $1 as options 
# to expr
#
while test `expr -- "$1" : "--.*"` -gt 0
do

  if test "$1" = "--nointbp"
  then
    NOINTBP=1
  elif test "$1" = "--mpicmd"
  then
    MPIRUN="$2"
    shift 1
  elif test "$1" = "--help"
  then
    echo "Usage: run_tests.sh [--mpicmd somecmd] [--nointbp] [install_dir]"
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



if test $NOINTBP -eq 1; then
  echo Not executing tests that use the interpolating backprojector
  echo
else
  echo Executing tests that use the interpolating backprojector
  echo If this is not what you want, rerun this script with the option --nointbp
  echo
fi

echo Not testing ecat file format conversion.
echo Run run_ecat_tests.sh separately for these tests.
echo





# first delete any files remaining from a previous run
rm -f my_*v my_*s

INSTALL_DIR=$1

command -v ${INSTALL_DIR}compare_image >/dev/null 2>&1 || { echo "${INSTALL_DIR}compare_image not found or not executable. Aborting." >&2; exit 1; }

ThereWereErrors=0

if test $NOINTBP = 0; then
echo
echo --------- TESTS THAT USE INTERPOLATING BACKPROJECTOR --------
echo
echo ------------- Running OSMAPOSL for sensitivity ------------- 
echo Running ${INSTALL_DIR}OSMAPOSL for sensitivity
${MPIRUN} ${INSTALL_DIR}OSMAPOSL OSMAPOSL_test_for_sensitivity.par 1> OSMAPOSL_test_for_sensitivity.log 2> OSMAPOSL_test_for_sensitivity_stderr.log 

echo '---- Comparing output of sensitivity (should be identical up to tolerance)'
echo Running ${INSTALL_DIR}compare_image
if ${INSTALL_DIR}compare_image RPTsens_seg4.hv my_RPTsens_seg4.hv;
then
echo ---- This test seems to be ok !;
else
echo There were problems here!;
ThereWereErrors=1;
fi

echo
echo ------------- Running OSMAPOSL ------------- 
echo Running ${INSTALL_DIR}OSMAPOSL
${MPIRUN} ${INSTALL_DIR}OSMAPOSL OSMAPOSL_test.par 1> OSMAPOSL_test.log 2> OSMAPOSL_test_stderr.log

echo '---- Comparing output of OSMAPOSL subiter 3 (should be identical up to tolerance)'
echo Running ${INSTALL_DIR}compare_image
if ${INSTALL_DIR}compare_image test_image_3.hv my_test_image_3.hv;
then
echo ---- This test seems to be ok !;
else
echo There were problems here!;
ThereWereErrors=1;
fi
echo '---- Comparing output of OSMAPOSL subiter 5 (should be identical up to tolerance)'
echo Running ${INSTALL_DIR}compare_image
if ${INSTALL_DIR}compare_image test_image_5.hv my_test_image_5.hv;
then
echo ---- This test seems to be ok !;
else
echo There were problems here!;
ThereWereErrors=1;
fi

fi # end of NOINTBP = 0

echo
echo --------- TESTS THAT USE PROJECTION MATRIX --------
echo
echo Generating initial image
# echo TODO check results
${INSTALL_DIR}generate_image generate_uniform_image.par
${INSTALL_DIR}postfilter my_uniform_image_circular.hv my_uniform_image.hv postfilter_truncate_circular_FOV.par
echo ------------- Running OSMAPOSL for sensitivity ------------- 
echo Running ${INSTALL_DIR}OSMAPOSL for sensitivity
${MPIRUN} ${INSTALL_DIR}OSMAPOSL OSMAPOSL_test_PM_for_sensitivity.par 1> sensitivity_PM.log 2> sensitivity_PM_stderr.log

echo '---- Comparing output of sensitivity (should be identical up to tolerance)'
echo Running ${INSTALL_DIR}compare_image
if ${INSTALL_DIR}compare_image RPTsens_seg3_PM.hv my_RPTsens_seg3_PM.hv;
then
echo ---- This test seems to be ok !;
else
echo There were problems here!;
ThereWereErrors=1;
fi

echo
echo -------- Running OSMAPOSL  with the MRP prior -------- 
echo Running ${INSTALL_DIR}OSMAPOSL
${MPIRUN} ${INSTALL_DIR}OSMAPOSL OSMAPOSL_test_PM_MRP.par 1> OSMAPOSL_PM_MRP.log 2> OSMAPOSL_PM_MRP_stderr.log

echo '---- Comparing output of OSMAPOSL subiter 6 (should be identical up to tolerance)'
echo Running ${INSTALL_DIR}compare_image
if ${INSTALL_DIR}compare_image test_image_PM_MRP_6.hv my_test_image_PM_MRP_6.hv;
then
echo ---- This test seems to be ok !;
else
echo There were problems here!;
ThereWereErrors=1;
fi

echo
echo -------- Running OSMAPOSL with a quadratic prior -------- 
echo Running ${INSTALL_DIR}OSMAPOSL
${MPIRUN} ${INSTALL_DIR}OSMAPOSL OSMAPOSL_test_PM_QP.par 1> OSMAPOSL_PM_QP.log 2> OSMAPOSL_PM_QP_stderr.log

echo '---- Comparing output of OSMAPOSL subiter 6 (should be identical up to tolerance)'
echo Running ${INSTALL_DIR}compare_image
if ${INSTALL_DIR}compare_image test_image_PM_QP_6.hv my_test_image_PM_QP_6.hv;
then
echo ---- This test seems to be ok !;
else
echo There were problems here!;
ThereWereErrors=1;
fi

echo
echo -------- Running OSMAPOSL with a quadratic prior with given weights -------- 
echo Running ${INSTALL_DIR}OSMAPOSL
${MPIRUN} ${INSTALL_DIR}OSMAPOSL OSMAPOSL_test_PM_QPweights.par 1> OSMAPOSL_PM_QPweights.log 2> OSMAPOSL_PM_QPweights_stderr.log

echo '---- Comparing output of OSMAPOSL subiter 6 (should be identical up to tolerance)'
echo Running ${INSTALL_DIR}compare_image
if ${INSTALL_DIR}compare_image test_image_PM_QPweights_6.hv my_test_image_PM_QPweights_6.hv;
then
echo ---- This test seems to be ok !;
else
echo There were problems here!;
ThereWereErrors=1;
fi

echo
echo -------- Writing ray tracing projection matrix to file for further checks -------
echo
if ${INSTALL_DIR}write_proj_matrix_by_bin  my_PMRT Utahscat600k_ca_seg4.hs write_proj_matrix_by_bin.par my_uniform_image_circular.hv 1> write_proj_matrix_by_bin.log 2> write_proj_matrix_by_bin_stderr.log;
then
echo ---- Projection matrix probably written ok!;
else
echo There were problems here!;
ThereWereErrors=1;
fi

echo
echo -------- Running OSMAPOSL stored projection matrix with a quadratic prior with given weights -------- 
echo Running ${INSTALL_DIR}OSMAPOSL
# Note: for this test, it is important that the projection matrix parameters in
# write_proj_matrix_by_bin.par and OSMAPOSL_test_PM_QPweights.par are the same.
${MPIRUN} ${INSTALL_DIR}OSMAPOSL OSMAPOSL_test_PMFromFile_QPweights.par 1> OSMAPOSL_PMFromFile_QPweights.log 2> OSMAPOSL_PMFromFile_QPweights_stderr.log

echo '---- Comparing output of OSMAPOSL subiter 6 (should be identical up to tolerance)'
echo Running ${INSTALL_DIR}compare_image
if ${INSTALL_DIR}compare_image test_image_PM_QPweights_6.hv my_test_image_PMFromFile_QPweights_6.hv;
then
echo ---- This test seems to be ok !;
else
echo There were problems here!;
ThereWereErrors=1;
fi

echo
echo -------- Running OSSPS with a quadratic prior -------- 
echo Running ${INSTALL_DIR}OSSPS
${MPIRUN} ${INSTALL_DIR}OSSPS OSSPS_test_PM_QP.par 1> OSSPS_PM_QP.log 2> OSSPS_PM_QP_stderr.log

echo '---- Comparing output of OSSPS subiter 8 (should be identical up to tolerance)'
echo Running ${INSTALL_DIR}compare_image
# relax test for the outer-rim voxels as these turn out to be more unstable than the internal ones
if ${INSTALL_DIR}compare_image -t 0.002 test_image_OSSPS_PM_QP_8.hv my_test_image_OSSPS_PM_QP_8.hv -a
   ${INSTALL_DIR}compare_image -r 1 test_image_OSSPS_PM_QP_8.hv my_test_image_OSSPS_PM_QP_8.hv
then
echo ---- This test seems to be ok !;
else
echo There were problems here!;
ThereWereErrors=1;
fi

echo
echo ------------- tests on stir_math and correct_projdata ---------
  echo "first make up some randoms (just a projdata full of 1)"
  ${INSTALL_DIR}stir_math -s --including-first \
         --times-scalar 0 --add-scalar 1 \
         my_fake_randoms Utahscat600k_ca_seg4.hs \
         1>stir_math_fake_randoms_stdout.log \
         2>stir_math_fake_randoms_stderr.log 
  echo "now make up a normalisation file (just projdata full of 2)"
  ${INSTALL_DIR}stir_math -s --including-first \
        --times-scalar 0 --add-scalar 2 \
        my_fake_norm Utahscat600k_ca_seg4.hs \
         1>stir_math_fake_norm_stdout.log \
         2>stir_math_fake_norm_stderr.log 
  echo "now run correct_projdata that will subtract randoms and then normalise"
  ${INSTALL_DIR}correct_projdata correct_projdata.par \
         1>correct_projdata_stdout.log \
         2>correct_projdata_stderr.log 
  echo "now do the same using stir_math"
  ${INSTALL_DIR}stir_math -s --times-scalar -1 \
	my_correct_projdata_test_rand \
	Utahscat600k_ca_seg4.hs my_fake_randoms.hs  \
         1>stir_math_do_randoms_stdout.log \
         2>stir_math_do_randoms_stderr.log 
  ${INSTALL_DIR}stir_math -s --mult \
	my_correct_projdata_test_check \
	my_correct_projdata_test_rand.hs  my_fake_norm.hs  \
         1>stir_math_do_norm_stdout.log \
         2>stir_math_do_norm_stderr.log 
  echo "finally, compare the 2 results. should be identical:"
  if ${INSTALL_DIR}compare_projdata  \
         my_correct_projdata_test_CR.hs \
	 my_correct_projdata_test_check.hs;
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



