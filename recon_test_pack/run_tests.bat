@echo off
rem Batch file for automatic running of the tests
rem see README.txt
rem Author: Kris Thielemans


echo This script should work with PARAPET software 0.91. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.

set INSTALL_DIR=%1

set ThereWereErrors=0
echo ------------- Converting ECAT6 file to Interfile ------------- 
%INSTALL_DIR%convecat6_if my_Utahscat600k_ca_seg4 Utahscat600k_ca.scn 1> convecat6_if.log 2> convecat6_if_stderr.log < convecat6_if.inp
echo ---- Comparing output of convecat6 (error should be 0)
%INSTALL_DIR%compare_projdata my_Utahscat600k_ca_seg4.hs Utahscat600k_ca_seg4.hs 2>compare_projdata_stderr.log
if ERRORLEVEL 1 goto conv_ecat6_problem
echo ---- This test seems to be ok !
goto run_sens
:conv_ecat6_problem
echo There were problems here!
set ThereWereErrors=1

:run_sens

echo.
echo ------------- Running sensitivity ------------- 
%INSTALL_DIR%sensitivity Utahscat600k_ca_seg4.hs 1> sensitivity.log 2> sensitivity_stderr.log < sensitivity.inp

echo ---- Comparing output of sensitivity (should be identical up to tolerance)
%INSTALL_DIR%compare_image RPTsens_seg4.hv my_RPTsens_seg4.hv
if ERRORLEVEL 1 goto sens_problem
echo ---- This test seems to be ok !
goto run_OSEM
:sens_problem
echo There were problems here!
set ThereWereErrors=1

:run_OSEM

echo.
echo ------------- Running OSEMMain ------------- 

%INSTALL_DIR%OSEMMain OSEMMain_test.par 1> OSEMMain.log 2> OSEMMain_stderr.log

echo ---- Comparing output of OSEMMain subiter 3 (should be identical up to tolerance)
%INSTALL_DIR%compare_image test_image_3.hv my_test_image_3.hv
if ERRORLEVEL 1 goto OSEM3_problem
echo ---- This test seems to be ok !
goto OSEM5
:OSEM3_problem
echo There were problems here!
set ThereWereErrors=1

:OSEM5

echo ---- Comparing output of OSEMMain subiter 3 (should be identical up to tolerance)
%INSTALL_DIR%compare_image test_image_5.hv my_test_image_5.hv
if ERRORLEVEL 1 goto OSEM5_problem
echo ---- This test seems to be ok !
goto the_end
:OSEM5_problem
echo There were problems here!
set ThereWereErrors=1

:the_end


echo.
echo --------------- End of tests -------------
echo.
if %ThereWereErrors%==1 echo Check what went wrong. The *.log files might help you.

if %ThereWereErrors%==0  echo Everything seems to be fine !




