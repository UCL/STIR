# Copyright 2011, Kris Thielemans
# Copyright (C) 2013-2014, University College London
# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file declaring all tests in this subdirectory

set(dir recon_test)

set(dir_SIMPLE_TEST_EXE_SOURCES ${dir}_SIMPLE_TEST_EXE_SOURCES)
set(dir_INVOLVED_TEST_EXE_SOURCES ${dir}_INVOLVED_TEST_EXE_SOURCES)
set(dir_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES ${dir}_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES)

set(${dir_SIMPLE_TEST_EXE_SOURCES}
	test_DataSymmetriesForBins_PET_CartesianGrid
        test_FBP2D
        test_FBP3DRP
        test_priors
)


set(${dir_INVOLVED_TEST_EXE_SOURCES}
        fwdtest
        bcktest
        recontest
        test_data_processor_projectors
        test_OSMAPOSL
)

include(stir_test_exe_targets)

# a test that uses MPI
create_stir_mpi_test(test_PoissonLogLikelihoodWithLinearModelForMeanAndProjData.cxx "${STIR_LIBRARIES}" "${STIR_REGISTRIES}")

# fwdtest and bcktest could be useful on their own, so we'll add them to the installation targets
if (BUILD_TESTING)
  install(TARGETS fwdtest bcktest DESTINATION bin)
endif()

# test_data_processor_projectors requires input argument
ADD_TEST(test_data_processor_projectors test_data_processor_projectors ${CMAKE_SOURCE_DIR}/recon_test_pack/Utahscat600k_ca_seg4.hs)


# test_OSMAPOSL can take input argument
ADD_TEST(test_OSMAPOSL_ray_tracing_matrix  test_OSMAPOSL)

if (parallelproj_FOUND)
  ADD_TEST(test_OSMAPOSL_parallelproj  test_OSMAPOSL ${CMAKE_SOURCE_DIR}/examples/samples/projector_pair_parallelproj.par)
endif()

