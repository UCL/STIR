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

set(dir test/numerics)

set(dir_SIMPLE_TEST_EXE_SOURCES ${dir}_SIMPLE_TEST_EXE_SOURCES)
set(dir_INVOLVED_TEST_EXE_SOURCES ${dir}_INVOLVED_TEST_EXE_SOURCES)

set(${dir_SIMPLE_TEST_EXE_SOURCES}
	test_matrices
	test_overlap_interpolate
	test_integrate_discrete_function
	test_IR_filters
	test_BSplines
	test_BSplinesRegularGrid1D
	test_BSplinesRegularGrid
	test_erf
)

# empty
set(${dir_INVOLVED_TEST_EXE_SOURCES}
)


include(../../stir_test_exe_targets.cmake)

