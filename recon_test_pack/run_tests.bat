@echo off
rem $Id$
rem Batch file for automatic running of the tests
rem see README.txt
rem Author: Kris Thielemans


echo This script should work with STIR version 1.1. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.
echo.

if NOT tmp%1==tmp--nointbp  set NOINTBP=0
if NOT tmp%1==tmp--nointbp  echo Executing tests that use the interpolating backprojector
if NOT tmp%1==tmp--nointbp  echo If this is not what you want, rerun this script with the option --nointbp


if tmp%1==tmp--nointbp  set NOINTBP=1
if tmp%1==tmp--nointbp  echo Not executing tests that use the interpolating backprojector
if tmp%1==tmp--nointbp  shift

echo.

set INSTALL_DIR=%1

set ThereWereErrors=0
echo ------------- Converting ECAT6 file to Interfile ------------- 
echo Running %INSTALL_DIR%convecat6_if
%INSTALL_DIR%convecat6_if my_Utahscat600k_ca_seg4 Utahscat600k_ca.scn 1> convecat6_if.log 2> convecat6_if_stderr.log < convecat6_if.inp
echo ---- Comparing output of convecat6 (error should be 0)
echo Running %INSTALL_DIR%compare_projdata 
%INSTALL_DIR%compare_projdata my_Utahscat600k_ca_seg4.hs Utahscat600k_ca_seg4.hs 2>compare_projdata_stderr.log
echo Running if 
if ERRORLEVEL 1 goto conv_ecat6_problem
echo ---- This test seems to be ok !
goto run_INTBP
:conv_ecat6_problem
echo There were problems here!
set ThereWereErrors=1


:run_INTBP

if %NOINTBP%==1 goto run_PM

echo.
echo --------- TESTS THAT USE INTERPOLATING BACKPROJECTOR --------
echo.
echo ------------- Running sensitivity ------------- 
echo Running %INSTALL_DIR%sensitivity 
%INSTALL_DIR%sensitivity OSMAPOSL_test_for_sensitivity.par 1> sensitivity.log 2> sensitivity_stderr.log < sensitivity.inp

echo ---- Comparing output of sensitivity (should be identical up to tolerance)
echo Running %INSTALL_DIR%compare_image 
%INSTALL_DIR%compare_image  RPTsens_seg4.hv my_RPTsens_seg4.hv
if ERRORLEVEL 1 goto sens_problem
echo ---- This test seems to be ok !
goto run_OSEM
:sens_problem
echo There were problems here!
set ThereWereErrors=1

:run_OSEM

echo.
echo ------------- Running OSMAPOSL ------------- 

echo Running %INSTALL_DIR%OSMAPOSL 
%INSTALL_DIR%OSMAPOSL OSMAPOSL_test.par 1> OSMAPOSL.log 2> OSMAPOSL_stderr.log

echo ---- Comparing output of OSMAPOSL subiter 3 (should be identical up to tolerance)
echo Running %INSTALL_DIR%compare_image 
%INSTALL_DIR%compare_image test_image_3.hv my_test_image_3.hv
if ERRORLEVEL 1 goto OSEM3_problem
echo ---- This test seems to be ok !
goto OSEM5
:OSEM3_problem
echo There were problems here!
set ThereWereErrors=1

:OSEM5

echo ---- Comparing output of OSMAPOSL subiter 5 (should be identical up to tolerance)
echo Running %INSTALL_DIR%compare_image 
%INSTALL_DIR%compare_image test_image_5.hv my_test_image_5.hv
if ERRORLEVEL 1 goto OSEM5_problem
echo ---- This test seems to be ok !
goto run_PM
:OSEM5_problem
echo There were problems here!
set ThereWereErrors=1



:run_PM
echo.
echo --------- TESTS THAT USE PROJECTION MATRIX --------
echo.
echo ------------- Running sensitivity ------------- 
echo Running %INSTALL_DIR%sensitivity 
%INSTALL_DIR%sensitivity OSMAPOSL_test_PM_for_sensitivity.par 1> sensitivity_PM.log 2> sensitivity_PM_stderr.log < sensitivity.inp

echo ---- Comparing output of sensitivity (should be identical up to tolerance)
echo Running %INSTALL_DIR%compare_image 
%INSTALL_DIR%compare_image  RPTsens_seg3_PM.hv my_RPTsens_seg3_PM.hv
if ERRORLEVEL 1 goto sensPM_problem
echo ---- This test seems to be ok !
goto run_OSEMPM
:sensPM_problem
echo There were problems here!
set ThereWereErrors=1

:run_OSEMPM

echo.
echo ------------- Running OSMAPOSL ------------- 

echo Running %INSTALL_DIR%OSMAPOSL 
%INSTALL_DIR%OSMAPOSL OSMAPOSL_test_PM_MRP.par 1> OSMAPOSL_PM_MRP.log 2> OSMAPOSL_PM_MRP_stderr.log


echo ---- Comparing output of OSMAPOSL subiter 6 (should be identical up to tolerance)
echo Running %INSTALL_DIR%compare_image 
%INSTALL_DIR%compare_image test_image_PM_MRP_6.hv my_test_image_PM_MRP_6.hv
if ERRORLEVEL 1 goto OSEMPM_problem
echo ---- This test seems to be ok !
goto the_end
:OSEMPM_problem
echo There were problems here!
set ThereWereErrors=1



:the_end


echo.
echo --------------- End of tests -------------
echo.
if %ThereWereErrors%==1 echo Check what went wrong. The *.log files might help you.

if %ThereWereErrors%==0  echo Everything seems to be fine !




