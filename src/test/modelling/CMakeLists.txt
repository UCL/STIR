#
#
# Copyright 2011-07-01 - 2011 Kris Thielemans
# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file declaring all tests in this subdirectory

set(dir test/modelling)

set(dir_SIMPLE_TEST_EXE_SOURCES ${dir}_SIMPLE_TEST_EXE_SOURCES)
set(dir_INVOLVED_TEST_EXE_SOURCES ${dir}_INVOLVED_TEST_EXE_SOURCES)

set(${dir_SIMPLE_TEST_EXE_SOURCES}
	test_ParametricDiscretisedDensity
)

set(${dir_INVOLVED_TEST_EXE_SOURCES}
	test_modelling
)

ADD_TEST(test_modelling
   ${CMAKE_CURRENT_BINARY_DIR}/test_modelling ${CMAKE_CURRENT_SOURCE_DIR}/input
)

include(stir_test_exe_targets)

