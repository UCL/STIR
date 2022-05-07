# Copyright 2011, Kris Thielemans
# Copyright (C) 2013-2014, University College London
# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file declaring all tests in this subdirectory and downloads the test data from zenodo

# A simple function used to get the Zenodo data
function(get_zenodo_file path url md5)
    message(STATUS "Downloading ${url} ...")
    file(DOWNLOAD "${url}" "${path}" EXPECTED_MD5 "${md5}")
endfunction()

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

if (DOWNLOAD_ZENODO_TEST_DATA)
    message(STATUS "Download Zenodo test data enabled.")
else()
    message(STATUS "Download Zenodo test data disabled.")
endif()

if (HAVE_CERN_ROOT)
    if (DOWNLOAD_ZENODO_TEST_DATA)
        ## DOWNLOAD THE ROOT CONSISTENCY TEST DATA FROM ZENODO (root file and corresponding header)
        ## TOTAL SIZE: ~3MB
        set(ROOT_STIR_consistency_data_dir ${CMAKE_SOURCE_DIR}/examples/ROOT_files/ROOT_STIR_consistency/pretest_output)
        set(ROOT_DATA_URL "https://zenodo.org/record/6410309/files")
        message(STATUS "Downloading ROOT consistency test data [~3MB] (if not already done)... ")


        # 1
        get_zenodo_file("${ROOT_STIR_consistency_data_dir}/root_data_test1.root"
                        "${ROOT_DATA_URL}/root_data_test1.root?download=1"
                        "0195985779dcd0cc238ba1cf73ec734b")
        get_zenodo_file("${ROOT_STIR_consistency_data_dir}/root_header_test1.hroot"
                        "${ROOT_DATA_URL}/root_header_test1.hroot?download=1"
                        "e1036f4cc527d6d4b05e47b059987eb7")
        # 2
        get_zenodo_file("${ROOT_STIR_consistency_data_dir}/root_data_test2.root"
                        "${ROOT_DATA_URL}/root_data_test2.root?download=1"
                        "93b20fb53fe7da16571839f99ea6788d")
        get_zenodo_file("${ROOT_STIR_consistency_data_dir}/root_header_test2.hroot"
                        "${ROOT_DATA_URL}/root_header_test2.hroot?download=1"
                        "ece24f080d81b32679e3ae95361d41dd")
        # 3
        get_zenodo_file("${ROOT_STIR_consistency_data_dir}/root_data_test3.root"
                    "${ROOT_DATA_URL}/root_data_test3.root?download=1"
                    "3d6d741d0709d8f0e034b63fba24eaa1")
        get_zenodo_file("${ROOT_STIR_consistency_data_dir}/root_header_test3.hroot"
                    "${ROOT_DATA_URL}/root_header_test3.hroot?download=1"
                    "9d4aa131344e887fbb15cd1004a8255e")
        # 4
        get_zenodo_file("${ROOT_STIR_consistency_data_dir}/root_data_test4.root"
                        "${ROOT_DATA_URL}/root_data_test4.root?download=1"
                        "c07d50333dbb2f42a7df6f3cc8c060ac")
        get_zenodo_file("${ROOT_STIR_consistency_data_dir}/root_header_test4.hroot"
                        "${ROOT_DATA_URL}/root_header_test4.hroot?download=1"
                        "4533e5d010abe190704546f79e1b5b8d")
        # 5
        get_zenodo_file("${ROOT_STIR_consistency_data_dir}/root_data_test5.root"
                        "${ROOT_DATA_URL}/root_data_test5.root?download=1"
                        "178955075c3d46cebd62b68c8b807643")
        get_zenodo_file("${ROOT_STIR_consistency_data_dir}/root_header_test5.hroot"
                        "${ROOT_DATA_URL}/root_header_test5.hroot?download=1"
                        "af8d2fe7a0a4668aaed2b2977096629c")
        # 6
        get_zenodo_file("${ROOT_STIR_consistency_data_dir}/root_data_test6.root"
                        "${ROOT_DATA_URL}/root_data_test6.root?download=1"
                        "e57b6668fa2cabf393523db85d0a994f")
        get_zenodo_file("${ROOT_STIR_consistency_data_dir}/root_header_test6.hroot"
                        "${ROOT_DATA_URL}/root_header_test6.hroot?download=1"
                        "6f6025593ef7891dbc4fa2a4964ad1f8")
        # 7
        get_zenodo_file("${ROOT_STIR_consistency_data_dir}/root_data_test7.root"
                        "${ROOT_DATA_URL}/root_data_test7.root?download=1"
                        "7b3faa81b3fe35f44e577cbab5b6a213")
        get_zenodo_file("${ROOT_STIR_consistency_data_dir}/root_header_test7.hroot"
                        "${ROOT_DATA_URL}/root_header_test7.hroot?download=1"
                        "174b9dfa3d88f9bbcaebc4c56e1ebf93")
        # 8
        get_zenodo_file("${ROOT_STIR_consistency_data_dir}/root_data_test8.root"
                        "${ROOT_DATA_URL}/root_data_test8.root?download=1"
                        "24150ce5d655212dc975989b3dbe4b33")
        get_zenodo_file("${ROOT_STIR_consistency_data_dir}/root_header_test8.hroot"
                        "${ROOT_DATA_URL}/root_header_test8.hroot?download=1"
                        "a821a3eac0000544f000cd5c74150023")
    endif()

    # Add test if the data is downloaded (only checking root_header_test1.hroot)
    if(EXISTS "${ROOT_STIR_consistency_data_dir}/root_header_test1.hroot")
        list(APPEND ${dir_INVOLVED_TEST_EXE_SOURCES} test_view_offset_root.cxx)
        set(test_name test_view_offset_root)
        # test_view_offset_root requires working directory to be examples/ROOT_files/ROOT_STIR_consistency
        # for output files to be written there
        ADD_TEST(NAME ${test_name}
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/examples/ROOT_files/ROOT_STIR_consistency
                COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_view_offset_root)
    endif()
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

