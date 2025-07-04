# Copyright (C) 2011, Kris Thielemans
# Copyright (C) 2013-2014, 2023 University College London

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake helper file, to be included by CMakeLists.txt files that
# set dir_SIMPLE_TEST_EXE_SOURCES, 
#     dir_INVOLVED_TEST_EXE_SOURCES_NO_REGISTRIES
# and dir_INVOLVED_TEST_EXE_SOURCES
# to a list of executables that need to be compiled (see below for tests that use MPI).
# Moreover, we will use ADD_TEST to create a test for
# the files in dir_SIMPLE_TEST_EXE_SOURCES and dir_INVOLVED_TEST_EXE_SOURCES_NO_REGISTRIES
# (assuming these don't need any parameters).
# You then have to use ADD_TEST for the executables in dir_INVOLVED_TEST_EXE_SOURCES
# 
#
# Alternatively, you can directly use one of 3 macros as follows:
#
#     create_stir_test(sometest.cxx "${STIR_LIBRARIES}" $<TARGET_OBJECTS:stir_registries>)
# without registries
#     create_stir_test(sometest.cxx "${STIR_LIBRARIES}" "")
# A test that uses any of the MPI routines (e.g. in the reconstruction library)
#     create_stir_mpi_test(sometest.cxx "${STIR_LIBRARIES}" $<TARGET_OBJECTS:stir_registries>)
# The above will execute the test with ${MPIEXEC_MAX_NUMPROCS} processors 
# (probably defaulting to 2 depending on your CMake version)
# A test for which you will use ADD_TEST yourself
#     create_stir_involved_test(sometest.cxx "${STIR_LIBRARIES}" $<TARGET_OBJECTS:stir_registries>)
# Note: from CMake 3.12, it is recommended to simply use stir_registries as opposed to $<TARGET_OBJECTS:stir_registries>,
# but we're backwards compatible for a while longer.

# Even more advanced usage:
#     create_stir_test (test_something.cxx "buildblock;IO;data_buildblock;recon_buildblock;buildblock" "")
# or when it needs to be linked with a registry (but this usage is not recommended as the registry mechanism is likely to change)
#     create_stir_test (test_something.cxx "buildblock" "${CMAKE_HOME_DIRECTORY}/buildblock/buildblock_registries.cxx")



#message(status dir_SIMPLE_TEST_EXE_SOURCES: ${dir_SIMPLE_TEST_EXE_SOURCES})
#message(status dir_SIMPLE_TEST_EXE_SOURCES: ${${dir_SIMPLE_TEST_EXE_SOURCES}})
#message(status dir_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES: ${dir_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES})
#message(status dir_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES: ${${dir_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES}})


if(NOT TARGET BUILD_TESTS)
  add_custom_target(BUILD_TESTS)
endif()

#### define macros
macro(add_STIR_CONFIG_DIR test_target)
  set_tests_properties(${test_target}
    PROPERTIES ENVIRONMENT "STIR_CONFIG_DIR=${PROJECT_SOURCE_DIR}/src/config")
endmacro()

macro (create_stir_involved_test source  libraries dependencies)
 if(BUILD_TESTING)
   get_filename_component(executable ${source} NAME_WE)
   add_executable(${executable} ${source} ${dependencies})
   target_link_libraries(${executable} PUBLIC ${libraries} PRIVATE fmt)
   SET_PROPERTY(TARGET ${executable} PROPERTY FOLDER "Tests")
   target_include_directories(${executable} PUBLIC ${Boost_INCLUDE_DIR})

   add_dependencies(BUILD_TESTS ${executable})
  endif()
endmacro( create_stir_involved_test)

macro (create_stir_test source  libraries dependencies)
 if(BUILD_TESTING)
   create_stir_involved_test(${source}  "${libraries}" "${dependencies}")
   ADD_TEST(${executable} ${CMAKE_CURRENT_BINARY_DIR}/${executable})
   add_STIR_CONFIG_DIR(${executable})
 endif()
endmacro( create_stir_test)

# for executables that use MPI.
macro (create_stir_mpi_test source  libraries dependencies)
 if(BUILD_TESTING)
   if(STIR_MPI)
     create_stir_involved_test(${source}  "${libraries}" "${dependencies}")
     ADD_TEST(${executable}  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS}  ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/${executable} ${MPIEXEC_POSTFLAGS})
     add_STIR_CONFIG_DIR(${executable})
   else()
     create_stir_test(${source}  "${libraries}" "${dependencies}")
   endif()
 endif()
endmacro( create_stir_mpi_test)


#### use the above macros for each target in dir_SIMPLE_TEST_EXE_SOURCES etc

foreach(executable ${${dir_SIMPLE_TEST_EXE_SOURCES}})
   create_stir_test (${executable} "${STIR_LIBRARIES}" $<TARGET_OBJECTS:stir_registries>)
endforeach()

# identical to above, but without including the registries as dependencies
foreach(executable ${${dir_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES}})
   create_stir_test (${executable} "${STIR_LIBRARIES}" "")
endforeach()

foreach(executable ${${dir_INVOLVED_TEST_EXE_SOURCES}})
   create_stir_involved_test (${executable} "${STIR_LIBRARIES}" $<TARGET_OBJECTS:stir_registries>)
endforeach()

