@rem $Id$: $Date$
@rem Batch programme to run PARAPET tests
@rem Kris Thielemans

@if #%1==# goto usage

@echo Running a sequence of non-interactive tests
@echo All tests should end with 'All tests ok !'
@echo There we go...
@pause
test_VectorWithOffset\%1\test_VectorWithOffset
@pause Press any key to continue to next test programme
test_Array\%1\test_Array
@pause 
test_convert_array\%1\test_convert_array
@pause 
test_IndexRange\%1\test_IndexRange
@pause 
test_filename_functions\%1\test_filename_functions
@pause 
test_linear_regression\%1\test_linear_regression  input\test_linear_regression.in
@pause 
test_coordinates\%1\test_coordinates
@pause 
test_filename_functions\%1\test_filename_functions 
@pause 
test_VoxelsOnCartesianGrid\%1\test_VoxelsOnCartesianGrid
@pause 
test_proj_data_info\%1\test_proj_data_info
@echo.
@echo End of tests !
@echo.

@goto exit

:usage
@echo Usage: run_tests configuration
@echo where the argument specifies the configuration name you want to test
@echo This will generally be Debug or Release
:exit


