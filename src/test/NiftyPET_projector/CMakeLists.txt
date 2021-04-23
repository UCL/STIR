# Copyright (C) 2019, University College London - Richard Brown
# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file declaring all tests in this subdirectory

set(dir test/NiftyPET_projector)

set(dir_SIMPLE_TEST_EXE_SOURCES ${dir}_SIMPLE_TEST_EXE_SOURCES)
set(dir_INVOLVED_TEST_EXE_SOURCES ${dir}_INVOLVED_TEST_EXE_SOURCES)
set(dir_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES ${dir}_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES)

set(${dir_INVOLVED_TEST_EXE_SOURCES}
	test_ProjectorNiftyPET_adjoint
)

ADD_TEST(test_ProjectorNiftyPET_adjoint
   ${CMAKE_CURRENT_BINARY_DIR}/test_ProjectorNiftyPET_adjoint 3)

include(stir_test_exe_targets)