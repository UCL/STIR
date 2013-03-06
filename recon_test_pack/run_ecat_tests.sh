#! /bin/sh
# Shell script for automatic running of the tests
# see README.txt
#
#  Copyright (C) 2001 PARAPET partners
#  Copyright (C) 2005- 2009-10-11, Hammersmith Imanet Ltd
#  Copyright (C) 2011-07-01 - $Date$, Kris Thielemans
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
# $Id$

echo This script should work with STIR version 2.3. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.
echo

#
# Parse option arguments (--)
# Note that the -- is required to suppress interpretation of $1 as options 
# to expr
#
while test `expr -- "$1" : "--.*"` -gt 0
do

if test "$1" = "--help"
  then
    echo "Usage: run_ecat_tests.sh [install_dir]"
    echo "See README.txt for more info."
    exit 1
  else
    echo Warning: Unknown option "$1"
    echo rerun with --help for more info.
    exit 1
  fi

  shift 1

done 



echo Executing tests on ecat file format conversion for projection data





# first delete any files remaining from a previous run
rm -f my_*v my_*s

INSTALL_DIR=$1

ThereWereErrors=0

run_ECAT6_tests=0
if [ $run_ECAT6_tests = 0 ]; then
echo ----------------- ECAT6 tests --------------------------------
echo No longer running ECAT6 tests as this file format is no longer supported.
else

echo ------------- Converting ECAT6 file to Interfile ------------- 
echo Running ${INSTALL_DIR}convecat6_if
${INSTALL_DIR}convecat6_if my_Utahscat600k_ca_seg4 Utahscat600k_ca.scn 1> convecat6_if.log 2> convecat6_if_stderr.log <  convecat6_if.inp
echo '---- Comparing output of convecat6_if (error should be 0)'
echo Running ${INSTALL_DIR}compare_projdata
if ${INSTALL_DIR}compare_projdata my_Utahscat600k_ca_seg4.hs Utahscat600k_ca_seg4.hs 2>compare_projdata_convecat6_if_stderr.log;
then
echo ---- This test seems to be ok !;
else
echo There were problems here!;
ThereWereErrors=1;
fi
fi

echo ------------- Converting Interfile to ECAT7 file ------------- 
echo Running ${INSTALL_DIR}conv_to_ecat7
${INSTALL_DIR}conv_to_ecat7 -s my_Utahscat600k_ca_seg4_ecat7.S Utahscat600k_ca_seg4.hs 1> conv_to_ecat7.log 2> conv_to_ecat7_stderr.log
echo '---- Comparing output of conv_to_ecat7 reading directly from ECAT7 (error should be 0)'
echo Running ${INSTALL_DIR}compare_projdata
if ${INSTALL_DIR}compare_projdata my_Utahscat600k_ca_seg4_ecat7.S Utahscat600k_ca_seg4.hs 2>compare_projdata_conv_to_ecat7_stderr.log;
then
echo ---- This test seems to be ok !;
else
echo There were problems here!;
ThereWereErrors=1;
fi


echo ----- Making Interfile headers for ECAT7
echo Running ${INSTALL_DIR}ifheaders_for_ecat7
${INSTALL_DIR}ifheaders_for_ecat7  my_Utahscat600k_ca_seg4_ecat7.S < /dev/null 1> ifheaders_for_ecat7.log 2> ifheaders_for_ecat7_stderr.log
echo '---- Comparing output of ifheaders_for_ecat7 on conv_to_ecat7 (error should be 0)'
echo Running ${INSTALL_DIR}compare_projdata
if ${INSTALL_DIR}compare_projdata my_Utahscat600k_ca_seg4_ecat7_S_f1g1d0b0.hs Utahscat600k_ca_seg4.hs 2>compare_projdata_ifheaders_for_ecat7_stderr.log;
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



