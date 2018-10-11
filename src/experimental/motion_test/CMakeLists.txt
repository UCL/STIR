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
set(dir motion_test)

set(dir_SIMPLE_TEST_EXE_SOURCES ${dir}_SIMPLE_TEST_EXE_SOURCES)
set(dir_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES ${dir}_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES)
set(dir_INVOLVED_TEST_EXE_SOURCES ${dir}_INVOLVED_TEST_EXE_SOURCES)

Set(${dir_INVOLVED_TEST_EXE_SOURCES}
    test_BSpline_transformations
    test_ProjectorsWithMotion
)

if (BUILD_TESTING)
	if (HAVE_ITK)
		ADD_TEST(test_BSpline_transformations 
			${CMAKE_CURRENT_BINARY_DIR}/test_BSpline_transformations 
				${CMAKE_CURRENT_SOURCE_DIR}/input/to_transform.nii.gz
				${CMAKE_CURRENT_SOURCE_DIR}/input/ground_truth.nii.gz
				${CMAKE_SOURCE_DIR}/examples/nifti/disp_4D.nii.gz)

		ADD_TEST(test_ProjectorsWithMotion 
			${CMAKE_CURRENT_BINARY_DIR}/test_ProjectorsWithMotion
				/Users/rich/Documents/Data/Kjell-patlak/Output/09_Movement/1_LR/dyn_im_LR_4.nii
				/Users/rich/Documents/Data/Kjell-patlak/Output/13_Prompts/1_LR/prompts_LR_4.hs
				/Users/rich/Documents/Data/Kjell-patlak/Output/10_DispFields/disp_LR.nii
				${CMAKE_CURRENT_SOURCE_DIR}/input/projector_pair.par)
	endif()
endif(BUILD_TESTING)

include(stir_test_exe_targets)