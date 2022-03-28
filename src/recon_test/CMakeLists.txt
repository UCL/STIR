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
	test_DataSymmetriesForBins_PET_CartesianGrid.cxx
        test_FBP2D.cxx
        test_FBP3DRP.cxx
        test_priors.cxx
        test_blocks_on_cylindrical_projectors.cxx
)


set(${dir_INVOLVED_TEST_EXE_SOURCES}
        fwdtest.cxx
        bcktest.cxx
        recontest.cxx
        test_data_processor_projectors.cxx
        test_OSMAPOSL.cxx
)

if (HAVE_CERN_ROOT)
    list(APPEND ${dir_INVOLVED_TEST_EXE_SOURCES} test_view_offset_root)
    foreach(n RANGE 1 12 1)
        set(test_name test_consistency_root_${n})
        ADD_TEST(NAME ${test_name}
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/examples/ROOT_files/ROOT_STIR_consistency
                COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_view_offset_root
                ${CMAKE_SOURCE_DIR}/examples/ROOT_files/ROOT_STIR_consistency
                )
    endforeach()
endif()

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

