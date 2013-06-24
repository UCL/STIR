#
# $Id$
#
# Copyright 2011-07-01 - $Date$ Kris Thielemans
# This file is part of STIR.
#
# This file is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
#
# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# See STIR/LICENSE.txt for details

# cmake file declaring all tests in the test subdirectory
set(dir test)

set(dir_SIMPLE_TEST_EXE_SOURCES ${dir}_SIMPLE_TEST_EXE_SOURCES)
set(dir_INVOLVED_TEST_EXE_SOURCES ${dir}_INVOLVED_TEST_EXE_SOURCES)

set(${dir_SIMPLE_TEST_EXE_SOURCES}
        test_Array
        test_NestedIterator
        test_VectorWithOffset
        test_ArrayFilter
        test_convert_array
	test_IndexRange
	test_coordinates
	test_filename_functions
	test_VoxelsOnCartesianGrid
	test_zoom_image
	test_proj_data_info
	test_ByteOrder
	test_Scanner
	test_ROIs
	test_ArcCorrection
	test_DynamicDiscretisedDensity  
	test_find_fwhm_in_image
        test_warp_image
)

set(${dir_INVOLVED_TEST_EXE_SOURCES}
	test_OutputFileFormat
	test_linear_regression
	test_stir_math
        # the next 2 are interactive, so we don't add a test for it, but only compile them
	test_display
	test_interpolate
)


include(../stir_test_exe_targets.cmake)

### add tests for the "involved" commands
### i.e. those that need command line arguments

ADD_TEST(test_linear_regression
   ${CMAKE_CURRENT_BINARY_DIR}/test_linear_regression ${CMAKE_CURRENT_SOURCE_DIR}/input/test_linear_regression.in
)

## test_stir_math needs to know the location of the stir_math executable
# Note that we cannot use get_target_property(var stir_math LOCATION) as it doesn't work for Visual Studio.
# Luckily, the following is simple enough (and in the cmake add_test documentation)
ADD_TEST(NAME test_stir_math
   COMMAND test_stir_math $<TARGET_FILE:stir_math>
)
# Final note: we could use this to avoid the use of ${CMAKE_CURRENT_BINARY_DIR} in the other tests, but both strategies work fine.

## add tests for OutputFileFormat

# we first construct a list of all input files
set(file_format_tests
	test_InterfileOutputFileFormat.in 
	test_InterfileOutputFileFormat_short.in 
)

if (HAVE_ECAT)
    #message("WARNING: ECAT6 tests currently disabled")
    list(APPEND file_format_tests
	test_ECAT7OutputFileFormat.in
    )
else(HAVE_ECAT)
    #message("No ECAT6/7 support compiled, so no tests for this file format")
endif(HAVE_ECAT)

# now for each of these, add a test
foreach(file_format ${file_format_tests})
  set(test_name test_OutputFileFormat_${file_format})
  ADD_TEST(${test_name}
    ${CMAKE_CURRENT_BINARY_DIR}/test_OutputFileFormat ${CMAKE_CURRENT_SOURCE_DIR}/input/${file_format}
)
endforeach()



