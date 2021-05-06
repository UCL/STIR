# Copyright (C) 2011, Kris Thielemans
# Copyright (C) 2013, University College London
# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
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
create_stir_test (test_Fourier.cxx "buildblock;IO;buildblock;numerics_buildblock;display" "")


include(stir_test_exe_targets)

