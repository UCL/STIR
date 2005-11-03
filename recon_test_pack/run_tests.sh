#! /bin/sh
# $Id$
# Shell script for automatic running of the tests
# see README.txt
# Author: Kris Thielemans

echo This script should work with STIR version 1.2 and 1.3. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.
echo

#
# Options
#
NOINTBP=0
DO_ECAT7_TESTS=0


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

  elif test "$1" = "--ecat7"
  then
    DO_ECAT7_TESTS=1;

  else
    echo Warning: Unknown option "$1"
    echo

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


if test $DO_ECAT7_TESTS -eq 1; then
  echo Executing tests on ecat7 file format conversion
  echo
else
  echo Not testing ecat7 file format conversion.
  echo If this is not what you want, rerun this script with the option --ecat7
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
if ${INSTALL_DIR}compare_projdata my_Utahscat600k_ca_seg4.hs Utahscat600k_ca_seg4.hs 2>compare_projdata_convecat6_if_stderr.log;
then
echo ---- This test seems to be ok !;
else
echo There were problems here!;
ThereWereErrors=1;
fi

if test $DO_ECAT7_TESTS=1; then

echo ------------- Converting Interfile to ECAT7 file ------------- 
echo Running ${INSTALL_DIR}conv_to_ecat7
${INSTALL_DIR}conv_to_ecat7 -s my_Utahscat600k_ca_seg4_ecat7.S Utahscat600k_ca_seg4.hs 1> conv_to_ecat7.log 2> conv_to_ecat7_stderr.log
echo '---- Comparing output of conv_to_ecat7 (error should be 0)'
echo Running ${INSTALL_DIR}compare_projdata
if ${INSTALL_DIR}compare_projdata my_Utahscat600k_ca_seg4_ecat7.S Utahscat600k_ca_seg4.hs 2>compare_projdata_conv_to_ecat7_stderr.log;
then
echo ---- This test seems to be ok !;
else
echo There were problems here!;
ThereWereErrors=1;
fi


echo ------------- Making Interfile headers for ECAT7 ------------- 
echo Running ${INSTALL_DIR}ifheaders_for_ecat7
${INSTALL_DIR}ifheaders_for_ecat7  my_Utahscat600k_ca_seg4_ecat7.S < /dev/null 1> ifheaders_for_ecat7.log 2> ifheaders_for_ecat7_stderr.log
echo '---- Comparing output of ifheaders_for_ecat7 (error should be 0)'
echo Running ${INSTALL_DIR}compare_projdata
if ${INSTALL_DIR}compare_projdata my_Utahscat600k_ca_seg4_ecat7_S_f1g1d0b0.hs Utahscat600k_ca_seg4.hs 2>compare_projdata_ifheaders_for_ecat7_stderr.log;
then
echo ---- This test seems to be ok !;
else
echo There were problems here!;
ThereWereErrors=1;
fi


echo ------------- Testing native reading of ECAT7 projdata ------------- 
echo '---- Comparing compare_projdata directly on ECAT7 sinograms to test STIR IO (error should be 0)'
echo Running ${INSTALL_DIR}compare_projdata
if ${INSTALL_DIR}compare_projdata my_Utahscat600k_ca_seg4_ecat7.S Utahscat600k_ca_seg4.hs 2>compare_projdata__ecat7_stderr.log;
then
echo ---- This test seems to be ok !;
else
echo There were problems here!;
ThereWereErrors=1;
fi

fi # end of ECAT7 tests


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
echo -------- Running OSMAPOSL  with the MRP prior -------- 
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
echo -------- Running OSMAPOSL with a quadratic prior -------- 
echo Running ${INSTALL_DIR}OSMAPOSL
${INSTALL_DIR}OSMAPOSL OSMAPOSL_test_PM_QP.par 1> OSMAPOSL_PM_QP.log 2> OSMAPOSL_PM_QP_stderr.log

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
${INSTALL_DIR}OSMAPOSL OSMAPOSL_test_PM_QPweights.par 1> OSMAPOSL_PM_QPweights.log 2> OSMAPOSL_PM_QPweights_stderr.log

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



