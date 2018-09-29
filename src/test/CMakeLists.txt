# Copyright (C) 2011-2012, Kris Thielemans
# Copyright (C) 2013, University College London
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
set(dir_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES ${dir}_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES)
set(dir_INVOLVED_TEST_EXE_SOURCES ${dir}_INVOLVED_TEST_EXE_SOURCES)

set(${dir_SIMPLE_TEST_EXE_SOURCES}
	test_ROIs
        test_warp_image
	test_DynamicDiscretisedDensity
)

set(${dir_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES}

)

Set(${dir_INVOLVED_TEST_EXE_SOURCES}
	test_OutputFileFormat
	IO/test_IO_DiscretisedDensity
	IO/test_IO_DynamicDiscretisedDensity
	IO/test_IO_ParametricDiscretisedDensity
	IO/test_IO_ITKMulticomponent
	test_linear_regression
	test_stir_math
        # the next 2 are interactive, so we don't add a test for it, but only compile them
	test_display
	test_interpolate
)

set(buildblock_simple_tests
        test_Array
        test_ArrayFilter
        test_NestedIterator
        test_VectorWithOffset
        test_convert_array
	test_IndexRange
	test_coordinates
	test_filename_functions
	test_VoxelsOnCartesianGrid
	test_zoom_image
	test_ByteOrder
	test_Scanner
	test_ArcCorrection
	test_find_fwhm_in_image
	test_proj_data_info
	test_proj_data_in_memory
	test_export_array
        test_GeneralisedPoissonNoiseGenerator
	test_multiple_proj_data
)

include(stir_test_exe_targets)

foreach(source ${buildblock_simple_tests})
  create_stir_test(${source} "buildblock;IO;buildblock;numerics_buildblock;display;IO" "")
endforeach()
#

### add tests for the "involved" commands
### i.e. those that need command line arguments

if (BUILD_TESTING)

ADD_TEST(test_linear_regression
   ${CMAKE_CURRENT_BINARY_DIR}/test_linear_regression ${CMAKE_CURRENT_SOURCE_DIR}/input/test_linear_regression.in
)

if (BUILD_EXECUTABLES)
## test_stir_math needs to know the location of the stir_math executable
# Note that we cannot use get_target_property(var stir_math LOCATION) as it doesn't work for Visual Studio.
# Luckily, the following is simple enough (and in the cmake add_test documentation)
ADD_TEST(NAME test_stir_math
   COMMAND test_stir_math $<TARGET_FILE:stir_math>
)
set_tests_properties(test_stir_math PROPERTIES DEPENDS stir_math)
# Final note: we could use TARGET_FILE to avoid the use of ${CMAKE_CURRENT_BINARY_DIR} in the other tests, but both strategies work fine.
endif()


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

if (HAVE_ITK)
    list(APPEND file_format_tests
	test_ITKDefaultOutputFileFormat.in
	test_ITKNiftiOutputFileFormat.in
    )
    ADD_TEST(test_IO_ITKMulticomponent
    	${CMAKE_CURRENT_BINARY_DIR}/test_IO_ITKMulticomponent ${CMAKE_SOURCE_DIR}/examples/nifti/disp_4D.nii)
endif()

# now for each of these, add a test
foreach(file_format ${file_format_tests})
	set(test_name test_IO_DiscretisedDensity_${file_format})
  ADD_TEST(${test_name}
		${CMAKE_CURRENT_BINARY_DIR}/test_IO_DiscretisedDensity ${CMAKE_CURRENT_SOURCE_DIR}/input/${file_format})
endforeach()
ADD_TEST(test_IO_ITKMulticomponent
${CMAKE_CURRENT_BINARY_DIR}/test_IO_ITKMulticomponent disp.nii)

# Parametric tests
ADD_TEST(test_IO_ParametricDiscretisedDensity_Interfile 
	${CMAKE_CURRENT_BINARY_DIR}/test_IO_ParametricDiscretisedDensity ${CMAKE_CURRENT_SOURCE_DIR}/input/test_InterfileOutputFileFormat.in)
ADD_TEST(test_IO_ParametricDiscretisedDensity_Interfile_short
	${CMAKE_CURRENT_BINARY_DIR}/test_IO_ParametricDiscretisedDensity ${CMAKE_CURRENT_SOURCE_DIR}/input/test_InterfileOutputFileFormat_short.in)

# Dynamic tests
ADD_TEST(test_IO_DynamicDiscretisedDensity_Interfile 
	${CMAKE_CURRENT_BINARY_DIR}/test_IO_DynamicDiscretisedDensity ${CMAKE_CURRENT_SOURCE_DIR}/input/test_InterfileOutputFileFormat.in)
ADD_TEST(test_IO_DynamicDiscretisedDensity_Interfile_short
	${CMAKE_CURRENT_BINARY_DIR}/test_IO_DynamicDiscretisedDensity ${CMAKE_CURRENT_SOURCE_DIR}/input/test_InterfileOutputFileFormat_short.in)
ADD_TEST(test_IO_DynamicDiscretisedDensity_Multi
	${CMAKE_CURRENT_BINARY_DIR}/test_IO_DynamicDiscretisedDensity ${CMAKE_CURRENT_SOURCE_DIR}/input/test_MultiOutputFileFormat.in)

endif(BUILD_TESTING)
