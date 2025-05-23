# Copyright (C) 2011-2012, Kris Thielemans
# Copyright (C) 2013, University College London
# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file declaring all tests in the test subdirectory
set(dir motion_test)

set(dir_SIMPLE_TEST_EXE_SOURCES ${dir}_SIMPLE_TEST_EXE_SOURCES)
set(dir_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES ${dir}_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES)
set(dir_INVOLVED_TEST_EXE_SOURCES ${dir}_INVOLVED_TEST_EXE_SOURCES)

Set(${dir_INVOLVED_TEST_EXE_SOURCES}
    test_BSpline_transformations.cxx
)

# create_stir_involved_test (test_BSpline_transformations.cxx "${STIR_LIBRARIES}" "${STIR_REGISTRIES}")

if (BUILD_TESTING)
	if (HAVE_ITK)
		ADD_TEST(test_BSpline_transformations 
			${CMAKE_CURRENT_BINARY_DIR}/test_BSpline_transformations 
				${CMAKE_CURRENT_SOURCE_DIR}/input/to_transform.nii.gz
				${CMAKE_CURRENT_SOURCE_DIR}/input/ground_truth.nii.gz
				${CMAKE_SOURCE_DIR}/examples/nifti/disp_4D.nii.gz)
	endif()
endif(BUILD_TESTING)

include(stir_test_exe_targets)
