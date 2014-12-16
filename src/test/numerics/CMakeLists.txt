# Copyright (C) 2011, Kris Thielemans
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

# cmake file declaring all tests in this subdirectory

set(dir test/numerics)

set(dir_SIMPLE_TEST_EXE_SOURCES ${dir}_SIMPLE_TEST_EXE_SOURCES)
set(dir_INVOLVED_TEST_EXE_SOURCES ${dir}_INVOLVED_TEST_EXE_SOURCES)
set(dir_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES ${dir}_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES)

create_stir_test (test_IR_filters.cxx "" "")
create_stir_test (test_BSplines.cxx "buildblock;IO;buildblock;numerics_buildblock;display" "")
create_stir_test (test_BSplinesRegularGrid.cxx "buildblock;IO;buildblock;numerics_buildblock;display" "")
create_stir_test (test_BSplinesRegularGrid1D.cxx "buildblock;IO;buildblock;numerics_buildblock;display" "")
create_stir_test (test_erf.cxx "" "")
create_stir_test (test_matrices.cxx "buildblock;IO;numerics_buildblock;buildblock;numerics_buildblock;display" "")
create_stir_test (test_overlap_interpolate.cxx "buildblock;IO;buildblock;numerics_buildblock;display" "")
create_stir_test (test_integrate_discrete_function.cxx "buildblock;IO;numerics_buildblock;display" "")


include(stir_test_exe_targets)

