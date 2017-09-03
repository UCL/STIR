@echo off
rem Batch file for automatic running of the tests
rem see README.txt
rem Author: Kris Thielemans


echo This script should work with STIR version 2.1 to 3.1. If you have
echo a later version, you might have to update your test pack.
echo Please check the web site.
echo.

if NOT tmp%1==tmp--nointbp  set NOINTBP=0
if NOT tmp%1==tmp--nointbp  echo Executing tests that use the interpolating backprojector
if NOT tmp%1==tmp--nointbp  echo If this is not what you want, rerun this script with the option --nointbp


if tmp%1==tmp--nointbp  set NOINTBP=1
if tmp%1==tmp--nointbp  echo Not executing tests that use the interpolating backprojector
if tmp%1==tmp--nointbp  shift

rem first delete any files remaining from a previous run
del my_*v 2> nul
del my_*s 2> nul

echo.

set INSTALL_DIR=%1

set ThereWereErrors=0

:run_INTBP

if %NOINTBP%==1 goto run_PM

echo.
echo --------- TESTS THAT USE INTERPOLATING BACKPROJECTOR --------
echo.
echo ------------- Running sensitivity ------------- 
echo Running %INSTALL_DIR%OSMAPOSL for sensitivity 
%INSTALL_DIR%OSMAPOSL OSMAPOSL_test_for_sensitivity.par 1> sensitivity.log 2> sensitivity_stderr.log


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
echo Generating initial image
rem TODO check results
%INSTALL_DIR%generate_image generate_uniform_image.par
%INSTALL_DIR%postfilter my_uniform_image_circular.hv my_uniform_image.hv postfilter_truncate_circular_FOV.par

echo.
echo ------------- Running sensitivity ------------- 
echo Running %INSTALL_DIR%OSMAPOSL for sensitivity 
%INSTALL_DIR%OSMAPOSL OSMAPOSL_test_PM_for_sensitivity.par 1> sensitivity_PM.log 2> sensitivity_PM_stderr.log

echo ---- Comparing output of sensitivity (should be identical up to tolerance)
echo Running %INSTALL_DIR%compare_image 
%INSTALL_DIR%compare_image  RPTsens_seg3_PM.hv my_RPTsens_seg3_PM.hv
if ERRORLEVEL 1 goto sensPM_problem
echo ---- This test seems to be ok !
goto run_OSEMPMMRP
:sensPM_problem
echo There were problems here!
set ThereWereErrors=1

:run_OSEMPMMRP

echo.
echo ----------- Running OSMAPOSL with MRP prior ------------- 

echo Running %INSTALL_DIR%OSMAPOSL 
%INSTALL_DIR%OSMAPOSL OSMAPOSL_test_PM_MRP.par 1> OSMAPOSL_PM_MRP.log 2> OSMAPOSL_PM_MRP_stderr.log


echo ---- Comparing output of OSMAPOSL subiter 6 (should be identical up to tolerance)
echo Running %INSTALL_DIR%compare_image 
%INSTALL_DIR%compare_image test_image_PM_MRP_6.hv my_test_image_PM_MRP_6.hv
if ERRORLEVEL 1 goto OSEMPM_problem
echo ---- This test seems to be ok !
goto run_OSEMPMQP
:OSEMPM_problem
echo There were problems here!
set ThereWereErrors=1

:run_OSEMPMQP

echo.
echo ----------- Running OSMAPOSL with Quadratic prior ------------- 

echo Running %INSTALL_DIR%OSMAPOSL 
%INSTALL_DIR%OSMAPOSL OSMAPOSL_test_PM_QP.par 1> OSMAPOSL_PM_QP.log 2> OSMAPOSL_PM_QP_stderr.log


echo ---- Comparing output of OSMAPOSL subiter 6 (should be identical up to tolerance)
echo Running %INSTALL_DIR%compare_image 
%INSTALL_DIR%compare_image test_image_PM_QP_6.hv my_test_image_PM_QP_6.hv
if ERRORLEVEL 1 goto OSEMQP_problem
echo ---- This test seems to be ok !
goto run_OSEMPMQPweights
:OSEMQP_problem
echo There were problems here!
set ThereWereErrors=1

:run_OSEMPMQPweights

echo.
echo ----------- Running OSMAPOSL with Quadratic prior ------------- 

echo Running %INSTALL_DIR%OSMAPOSL 
%INSTALL_DIR%OSMAPOSL OSMAPOSL_test_PM_QPweights.par 1> OSMAPOSL_PM_QPweights.log 2> OSMAPOSL_PM_QPweights_stderr.log


echo ---- Comparing output of OSMAPOSL subiter 6 (should be identical up to tolerance)
echo Running %INSTALL_DIR%compare_image 
%INSTALL_DIR%compare_image test_image_PM_QPweights_6.hv my_test_image_PM_QPweights_6.hv
if ERRORLEVEL 1 goto OSEMQPweights_problem
echo ---- This test seems to be ok !
goto run_OSSPS_PM_QP
:OSEMQPweights_problem
echo There were problems here!
set ThereWereErrors=1


:run_OSSPS_PM_QP

echo.
echo ----------- Running OSSPS with Quadratic prior ------------- 

echo Running %INSTALL_DIR%OSSPS 
%INSTALL_DIR%OSSPS OSSPS_test_PM_QP.par 1> OSSPS_PM_QP.log 2> OSSPS_PM_QP_stderr.log


echo ---- Comparing output of OSSPS subiter 8 (should be identical up to tolerance)
echo Running %INSTALL_DIR%compare_image 
%INSTALL_DIR%compare_image test_image_OSSPS_PM_QP_8.hv my_test_image_OSSPS_PM_QP_8.hv
if ERRORLEVEL 1 goto OSSPSQP_problem
echo ---- This test seems to be ok !
goto run_CORRECT_PROJDATA
:OSSPSQP_problem
echo There were problems here!
set ThereWereErrors=1

:run_CORRECT_PROJDATA
echo.
echo ------------- tests on stir_math and correct_projdata ---------
  echo first make up some randoms (just a projdata full of 1)
  %INSTALL_DIR%stir_math -s --including-first --times-scalar 0 --add-scalar 1   my_fake_randoms Utahscat600k_ca_seg4.hs   1>stir_math_fake_randoms_stdout.log   2>stir_math_fake_randoms_stderr.log 
  echo now make up a normalisation file (just projdata full of 2)
  %INSTALL_DIR%stir_math -s --including-first --times-scalar 0 --add-scalar 2   my_fake_norm Utahscat600k_ca_seg4.hs   1>stir_math_fake_norm_stdout.log   2>stir_math_fake_norm_stderr.log 
  echo now run correct_projdata that will subtract randoms and then normalise
  %INSTALL_DIR%correct_projdata correct_projdata.par   1>correct_projdata_stdout.log   2>correct_projdata_stderr.log 
  echo now do the same using stir_math
  %INSTALL_DIR%stir_math -s --times-scalar -1	my_correct_projdata_test_rand	Utahscat600k_ca_seg4.hs my_fake_randoms.hs    1>stir_math_do_randoms_stdout.log   2>stir_math_do_randoms_stderr.log 
  %INSTALL_DIR%stir_math -s --mult	my_correct_projdata_test_check	my_correct_projdata_test_rand.hs  my_fake_norm.hs    1>stir_math_do_norm_stdout.log   2>stir_math_do_norm_stderr.log 
  echo finally, compare the 2 results. should be identical:
  %INSTALL_DIR%compare_projdata    my_correct_projdata_test_CR.hs 	 my_correct_projdata_test_check.hs
  if ERRORLEVEL 1 goto CORRECT_PROJDATA_problem
  echo ---- This test seems to be ok !
  goto the_end
:CORRECT_PROJDATA_problem
echo There were problems here!
set ThereWereErrors=1


:the_end


echo.
echo --------------- End of tests -------------
echo.
if %ThereWereErrors%==1 echo Check what went wrong. The *.log files might help you.

if %ThereWereErrors%==0  echo Everything seems to be fine !




