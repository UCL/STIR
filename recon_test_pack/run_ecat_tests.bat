@echo off
rem $Id$
rem Batch file for automatic running of the tests
rem see README.txt
rem Author: Kris Thielemans


echo This script should work with STIR version 2.3. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.
echo.

rem first delete any files remaining from a previous run
del my_*v 2> nul
del my_*s 2> nul

echo.

set INSTALL_DIR=%1

set ThereWereErrors=0

:run_conv_to_ecat7

echo ------------- Converting Interfile to ECAT7 file ------------- 
echo Running %INSTALL_DIR%conv_to_ecat7
%INSTALL_DIR%conv_to_ecat7 -s my_Utahscat600k_ca_seg4_ecat7.S Utahscat600k_ca_seg4.hs 1> conv_to_ecat7.log 2> conv_to_ecat7_stderr.log
echo ---- Comparing output of conv_to_ecat7 (error should be 0)
echo Running %INSTALL_DIR%compare_projdata
%INSTALL_DIR%compare_projdata my_Utahscat600k_ca_seg4_ecat7.S Utahscat600k_ca_seg4.hs 2>compare_projdata_conv_to_ecat7_stderr.log
if ERRORLEVEL 1 goto conv_to_ecat7_problem
echo ---- This test seems to be ok !
goto run_ifheaders_for_ecat7
:conv_to_ecat7_problem
echo There were problems here!
set ThereWereErrors=1


:run_ifheaders_for_ecat7

echo ------------- Making Interfile headers for ECAT7 ------------- 
echo Running %INSTALL_DIR%ifheaders_for_ecat7
%INSTALL_DIR%ifheaders_for_ecat7  my_Utahscat600k_ca_seg4_ecat7.S < NUL: 1> ifheaders_for_ecat7.log 2> ifheaders_for_ecat7_stderr.log
echo ---- Comparing output of ifheaders_for_ecat7 (error should be 0)
echo Running %INSTALL_DIR%compare_projdata
%INSTALL_DIR%compare_projdata my_Utahscat600k_ca_seg4_ecat7_S_f1g1d0b0.hs Utahscat600k_ca_seg4.hs 2>compare_projdata_ifheaders_for_ecat7_stderr.log
if ERRORLEVEL 1 goto ifheaders_for_ecat7_problem
echo ---- This test seems to be ok !
goto run_native_ecat7_projdata
:ifheaders_for_ecat7_problem
echo There were problems here!
set ThereWereErrors=1

:run_native_ecat7_projdata

echo ------------- Testing native reading of ECAT7 projdata ------------- 
echo ---- Comparing compare_projdata directly on ECAT7 sinograms to test STIR IO (error should be 0)
echo Running %INSTALL_DIR%compare_projdata
%INSTALL_DIR%compare_projdata my_Utahscat600k_ca_seg4_ecat7.S Utahscat600k_ca_seg4.hs 2>compare_projdata__ecat7_stderr.log
if ERRORLEVEL 1 goto the_end
echo ---- This test seems to be ok !
goto the_end
echo There were problems here!
set ThereWereErrors=1



:the_end


echo.
echo --------------- End of tests -------------
echo.
if %ThereWereErrors%==1 echo Check what went wrong. The *.log files might help you.

if %ThereWereErrors%==0  echo Everything seems to be fine !




