#! /bin/sh
# $Id$
# Shell script for automatic running of the tests
# see README.txt
# Author: Kris Thielemans

echo This script should work with STIR version 1.2. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.
echo

if test tmp$1 = tmp--nointbp; then
  NOINTBP=1
  echo Not executing tests that use the interpolating backprojector
  echo
  shift
else
  NOINTBP=0
  echo Executing tests that use the interpolating backprojector
  echo If this is not what you want, rerun this script with the option --nointbp
  echo
fi

# first delete any files remaining from a previous run
rm -f my_*v my_*s

INSTALL_DIR=$1

ThereWereErrors=0
echo ------------- Converting ECAT6 file to Interfile ------------- 
echo Running ${INSTALL_DIR}convecat6_if
${INSTALL_DIR}convecat6_if my_Utahscat600k_ca_seg4 Utahscat600k_ca.scn 1> convecat6_if.log 2> convecat6_if_stderr.log <  convecat6_if.inp
echo '---- Comparing output of convecat6 (error should be 0)'
echo Running ${INSTALL_DIR}compare_projdata
if ${INSTALL_DIR}compare_projdata my_Utahscat600k_ca_seg4.hs Utahscat600k_ca_seg4.hs 2>compare_projdata_stderr.log;
then
echo ---- This test seems to be ok !;
else
echo There were problems here!;
ThereWereErrors=1;
fi

if test $NOINTBP = 0; then
echo
echo --------- TESTS THAT USE INTERPOLATING BACKPROJECTOR --------
echo
echo ------------- Running sensitivity ------------- 
echo Running ${INSTALL_DIR}sensitivity 
${INSTALL_DIR}sensitivity OSMAPOSL_test_for_sensitivity.par 1> sensitivity.log 2> sensitivity_stderr.log < sensitivity.inp

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
${INSTALL_DIR}OSMAPOSL OSMAPOSL_test.par 1> OSMAPOSL.log 2> OSMAPOSL_stderr.log

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
echo ------------- Running sensitivity ------------- 
echo Running ${INSTALL_DIR}sensitivity 
${INSTALL_DIR}sensitivity OSMAPOSL_test_PM_for_sensitivity.par 1> sensitivity_PM.log 2> sensitivity_PM_stderr.log < sensitivity.inp

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
echo ------------- Running OSMAPOSL ------------- 
echo Running ${INSTALL_DIR}OSMAPOSL
${INSTALL_DIR}OSMAPOSL OSMAPOSL_test_PM_MRP.par 1> OSMAPOSL_PM_MRP.log 2> OSMAPOSL_PM_MRP_stderr.log

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
echo 'You could remove all generated files using "rm my_* *.log"'
fi



