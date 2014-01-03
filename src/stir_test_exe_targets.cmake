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

# cmake helper file, to be included by CMakeLists.txt files that
# set dir_SIMPLE_TEST_EXE_SOURCES, 
#     dir_INVOLVED_TEST_EXE_SOURCES_NO_REGISTRIES
# and dir_INVOLVED_TEST_EXE_SOURCES
# to a list of executables that need to be compiled.
# Moreover, we will use ADD_TEST to create a test for
# the files in dir_SIMPLE_TEST_EXE_SOURCES and dir_INVOLVED_TEST_EXE_SOURCES_NO_REGISTRIES
# (assuming these don't need any parameters).
# The user has to add ADD_TEST for the executables in dir_INVOLVED_TEST_EXE_SOURCES
#
# Alternatively, the user can directly use the macros create_stir_involved_test and create_stir_test as follows:
#
#   create_stir_test (test_something.cxx "buildblock;recon_buildblock" "")
# or when it needs to be linked with a registry (but this usage is not recommended as the registry mechanism is likely to change)
#   create_stir_test (test_something.cxx "buildblock;recon_buildblock" "${CMAKE_HOME_DIRECTORY}/buildblock/buildblock_registries.cxx")



#message(status dir_SIMPLE_TEST_EXE_SOURCES: ${dir_SIMPLE_TEST_EXE_SOURCES})
#message(status dir_SIMPLE_TEST_EXE_SOURCES: ${${dir_SIMPLE_TEST_EXE_SOURCES}})
#message(status dir_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES: ${dir_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES})
#message(status dir_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES: ${${dir_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES}})

macro (create_stir_involved_test source  libraries dependencies)
   get_filename_component(executable ${source} NAME_WE)
   add_executable(${executable} ${source} ${dependencies})
   target_link_libraries(${executable} ${libraries})
   SET_PROPERTY(TARGET ${executable} PROPERTY FOLDER "Tests")
endmacro( create_stir_involved_test)

macro (create_stir_test source  libraries dependencies)
   create_stir_involved_test(${source}  "${libraries}" "${dependencies}")
   ADD_TEST(${executable} ${CMAKE_CURRENT_BINARY_DIR}/${executable})
endmacro( create_stir_test)


foreach(executable ${${dir_SIMPLE_TEST_EXE_SOURCES}})
   create_stir_test (${executable}.cxx "${STIR_LIBRARIES}" "${STIR_REGISTRIES}")
endforeach()

# identical to above, but without including the registries as dependencies
foreach(executable ${${dir_SIMPLE_TEST_EXE_SOURCES_NO_REGISTRIES}})
   create_stir_test (${executable}.cxx "${STIR_LIBRARIES}" "")
endforeach()

foreach(executable ${${dir_INVOLVED_TEST_EXE_SOURCES}})
   create_stir_involved_test (${executable}.cxx "${STIR_LIBRARIES}" "${STIR_REGISTRIES}")
endforeach()

