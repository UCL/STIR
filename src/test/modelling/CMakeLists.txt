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

include(../../stir_test_exe_targets.cmake)

