# Copyright (C) 2011-2012, Kris Thielemans
# Copyright (C) 2013-2022 University College London
# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file declaring all tests in the test subdirectory
set(dir test)

set(dir_SIMPLE_TEST_EXE_SOURCES ${dir}_SIMPLE_TEST_EXE_SOURCES)
set(dir_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES ${dir}_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES)
set(dir_INVOLVED_TEST_EXE_SOURCES ${dir}_INVOLVED_TEST_EXE_SOURCES)

set(${dir_SIMPLE_TEST_EXE_SOURCES}
	test_ROIs.cxx
        test_warp_image.cxx
	test_DynamicDiscretisedDensity.cxx
	test_ScatterSimulation.cxx
        test_ML_norm.cxx
	test_proj_data_info_subsets.cxx
)

set(${dir_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES}
        test_DateTime.cxx
        test_radionuclide.cxx
)

Set(${dir_INVOLVED_TEST_EXE_SOURCES}
	test_OutputFileFormat.cxx
	IO/test_IO_DiscretisedDensity.cxx
	IO/test_IO_DynamicDiscretisedDensity.cxx
	IO/test_IO_ParametricDiscretisedDensity.cxx
	IO/test_IO_ITKMulticomponent.cxx
	test_linear_regression.cxx
	test_stir_math.cxx
        # the next 2 are interactive, so we don't add a test for it, but only compile them
	test_display.cxx
	test_interpolate.cxx
)

set(buildblock_simple_tests
        test_Array.cxx
        test_ArrayFilter.cxx
        test_SeparableMetzArrayFilter.cxx
        test_SeparableGaussianArrayFilter.cxx
        test_NestedIterator.cxx
        test_VectorWithOffset.cxx
        test_convert_array.cxx
	test_IndexRange.cxx
	test_coordinates.cxx
	test_filename_functions.cxx
        test_KeyParser.cxx
	test_VoxelsOnCartesianGrid.cxx
	test_zoom_image.cxx
	test_ByteOrder.cxx
        test_ImagingModality.cxx
	test_Scanner.cxx
	test_ArcCorrection.cxx
	test_find_fwhm_in_image.cxx
        test_DetectionPosition.cxx
	test_proj_data_info.cxx
	test_DetectorCoordinateMap.cxx
	test_proj_data.cxx
	test_proj_data_maths.cxx
	test_export_array.cxx
        test_GeneralisedPoissonNoiseGenerator.cxx
	test_multiple_proj_data.cxx
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
    	${CMAKE_CURRENT_BINARY_DIR}/test_IO_ITKMulticomponent ${CMAKE_SOURCE_DIR}/examples/nifti/disp_4D.nii.gz)
endif()

# now for each of these, add a test
foreach(file_format ${file_format_tests})
	set(test_name test_IO_DiscretisedDensity_${file_format})
  ADD_TEST(${test_name}
		${CMAKE_CURRENT_BINARY_DIR}/test_IO_DiscretisedDensity ${CMAKE_CURRENT_SOURCE_DIR}/input/${file_format})
endforeach()

# Parametric tests
ADD_TEST(test_IO_ParametricDiscretisedDensity_Interfile 
	${CMAKE_CURRENT_BINARY_DIR}/test_IO_ParametricDiscretisedDensity ${CMAKE_CURRENT_SOURCE_DIR}/input/test_InterfileOutputFileFormat.in)
ADD_TEST(test_IO_ParametricDiscretisedDensity_Interfile_short
	${CMAKE_CURRENT_BINARY_DIR}/test_IO_ParametricDiscretisedDensity ${CMAKE_CURRENT_SOURCE_DIR}/input/test_InterfileOutputFileFormat_short.in)
ADD_TEST(test_IO_ParametricDiscretisedDensity_Multi
	${CMAKE_CURRENT_BINARY_DIR}/test_IO_ParametricDiscretisedDensity ${CMAKE_CURRENT_SOURCE_DIR}/input/test_MultiOutputFileFormat.in)

# Dynamic tests
ADD_TEST(test_IO_DynamicDiscretisedDensity_Interfile 
	${CMAKE_CURRENT_BINARY_DIR}/test_IO_DynamicDiscretisedDensity ${CMAKE_CURRENT_SOURCE_DIR}/input/test_InterfileOutputFileFormat.in)
ADD_TEST(test_IO_DynamicDiscretisedDensity_Interfile_short
	${CMAKE_CURRENT_BINARY_DIR}/test_IO_DynamicDiscretisedDensity ${CMAKE_CURRENT_SOURCE_DIR}/input/test_InterfileOutputFileFormat_short.in)
ADD_TEST(test_IO_DynamicDiscretisedDensity_Multi
	${CMAKE_CURRENT_BINARY_DIR}/test_IO_DynamicDiscretisedDensity ${CMAKE_CURRENT_SOURCE_DIR}/input/test_MultiOutputFileFormat.in)

endif(BUILD_TESTING)
