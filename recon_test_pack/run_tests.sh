#! /bin/sh
# Shell script for automatic running of the tests
# see README.txt
# Author: Kris Thielemans

echo This script should work with PARAPET software 0.93. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.

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

echo
echo ------------- Running sensitivity ------------- 
echo Running ${INSTALL_DIR}sensitivity
${INSTALL_DIR}sensitivity Utahscat600k_ca_seg4.hs 1> sensitivity.log 2> sensitivity_stderr.log < sensitivity.inp

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


echo
echo '--------------- End of tests -------------'
echo
if test ${ThereWereErrors} = 1  ; 
then
echo "Check what went wrong. The *.log files might help you."
else
echo "Everything seems to be fine !"
fi



