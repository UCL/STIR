@rem Batch programme to run STIR tests
@rem Kris Thielemans

@if #%1==# goto usage

@echo Running a sequence of non-interactive tests
@echo All tests should end with 'All tests ok !'
@echo There we go...
@pause
VC\test_VectorWithOffset\%1\test_VectorWithOffset
@pause Press any key to continue to next test programme
VC\test_ArcCorrection\%1\test_ArcCorrection
@pause Press any key to continue to next test programme
VC\test_Array\%1\test_Array
@pause
VC\test_convert_array\%1\test_convert_array
@pause 
VC\test_IndexRange\%1\test_IndexRange
@pause 
VC\test_coordinates\%1\test_coordinates
@pause 
VC\test_ArrayFilter\%1\test_ArrayFilter
@pause
VC\test_NestedIterator\%1\test_NestedIterator
@pause
VC\test_filename_functions\%1\test_filename_functions
@pause 
VC\test_linear_regression\%1\test_linear_regression  input\test_linear_regression.in
rem next test is interactive and disabled
rem @pause 
rem VC\test_interpolate\%1\test_interpolate
@pause 
VC\test_VoxelsOnCartesianGrid\%1\test_VoxelsOnCartesianGrid
@pause 
VC\test_DynamicDiscretisedDensity\%1\test_DynamicDiscretisedDensity
@pause 
VC\test_zoom_image\%1\test_zoom_image
@pause 
VC\test_proj_data_info\%1\test_proj_data_info

@pause 
VC\test_OutputFileFormat\%1\test_OutputFileFormat input/test_InterfileOutputFileFormat.in 
@pause 
VC\test_OutputFileFormat\%1\test_OutputFileFormat input/test_InterfileOutputFileFormat_short.in 
@pause 
VC\test_OutputFileFormat\%1\test_OutputFileFormat input/test_ECAT6OutputFileFormat.in
@pause 
VC\test_OutputFileFormat\%1\test_OutputFileFormat input/test_ECAT7OutputFileFormat.in
@pause 
VC\test_ByteOrder\%1\test_ByteOrder
@pause
VC\test_ROIs\%1\test_ROIs
@pause
VC\test_Scanner\%1\test_Scanner
@pause
rem next test disabled
rem VC\test_VAXfloat\%1\test_VAXfloat --read input\test_VAXfloat.in
@pause 
@echo Executing test_stir_math 
@echo This relies on having stir_math compiled (in the utilities workspace).
VC\test_stir_math\%1\test_stir_math ..\utilities\VC\stir_math\%1\stir_math 
@echo.
@echo End of tests !
@echo.

@goto exit

:usage
@echo Usage: run_tests configuration
@echo where the argument specifies the configuration name you want to test
@echo This will generally be Debug or Release
:exit


